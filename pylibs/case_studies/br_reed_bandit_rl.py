#!/usr/bin/env python3
# Copyright (c) 2026The OTNS Authors.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the
#    names of its contributors may be used to endorse or promote products
#    derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import argparse
import csv
import os
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from otns.cli import OTNS
from otns.cli.errors import OTNSExitedError


SIMULATION_TIME = 3600.0
FORMATION_TIME = 120.0
GRID_SPACING_M = 20
METER_PER_UNIT = 1.0
RADIO_RANGE_M = 55
BR_POS = (40, 0)
REED_ACTIONS = (0.5, 0.8, 1.0, 1.2, 1.5)
INITIAL_TX_PERIOD = 2.0
COAP_PAYLOAD_BYTES = 100
EPSILON = 0.50
RANDOM_SEED = 1
TIME_EPSILON_S = 1e-6
MIN_GO_DURATION_S = 0.1
DECISION_INTERVAL = 10.0


@dataclass
class BufferInfo:
    total: Optional[int] = None
    free: Optional[int] = None
    raw: str = ""

    @property
    def normalized_free(self) -> float:
        if self.total is None or self.free is None or self.total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.free / self.total))


@dataclass
class ContextualEpsilonGreedyBandit:
    actions: Tuple[float, ...]
    epsilon: float
    rng: random.Random
    alpha: float = 0.1
    epsilon_decay: float = 0.99
    min_epsilon: float = 0.01
    q_values: Dict[Tuple[int, int, int, int], List[float]] = field(default_factory=dict)
    counts: Dict[Tuple[int, int, int, int], List[int]] = field(default_factory=dict)

    def select(self, state: Tuple[int, int, int, int]) -> Tuple[float, int, bool]:
        self._ensure_state(state)
        if self.rng.random() < self.epsilon:
            action_index = self.rng.randrange(len(self.actions))
            explored = True
        else:
            action_index = max(range(len(self.actions)), key=lambda i: self.q_values[state][i])
            explored = False
            
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        return self.actions[action_index], action_index, explored

    def update(self, state: Tuple[int, int, int, int], action_index: int, reward: float) -> None:
        self._ensure_state(state)
        self.counts[state][action_index] += 1
        old_q = self.q_values[state][action_index]
        self.q_values[state][action_index] = old_q + self.alpha * (reward - old_q)

    def _ensure_state(self, state: Tuple[int, int, int, int]) -> None:
        if state in self.q_values:
            return
        self.q_values[state] = [0.0 for _ in self.actions]
        self.counts[state] = [0 for _ in self.actions]


APP_MESSAGE_PERIOD = 2.0
MAX_APP_QUEUE_SIZE = 10


@dataclass
class ReedRuntime:
    node_id: int
    name: str
    bandit: ContextualEpsilonGreedyBandit
    tx_window_start: float = 0.0
    tx_period: float = INITIAL_TX_PERIOD
    tx_random_offset: float = 0.0
    next_tx_time: float = 0.0
    tx_count: int = 0
    rx_count: int = 0
    app_queue: int = 0
    last_app_gen_time: float = 0.0

    app_drop_count: int = 0
    last_app_drop_count: int = 0
    last_state: Optional[Tuple[int, int, int, int]] = None
    last_action_index: Optional[int] = None
    last_tx_err_cca: int = 0


def parse_bufferinfo(lines: List[str]) -> BufferInfo:
    raw = " | ".join(line.strip() for line in lines)
    info = BufferInfo(raw=raw)

    for line in lines:
        lower = line.lower()
        numbers = [int(num) for num in re.findall(r"\d+", line)]
        if not numbers:
            continue
        if re.search(r"\btotal\b", lower):
            info.total = numbers[-1]
        if re.search(r"\bfree\b", lower):
            info.free = numbers[-1]

    # Some OpenThread versions show a table with columns such as:
    # | total: 40 | free: 36 | or "total 40 free 36".
    total_free = re.search(r"total\D+(\d+).*free\D+(\d+)", raw, re.IGNORECASE)
    if total_free:
        info.total = int(total_free.group(1))
        info.free = int(total_free.group(2))

    return info


def free_queue_bucket(info: BufferInfo) -> int:
    if info.free is None:
        return 0
    if info.free < 10:
        return 0
    if info.free < 20:
        return 1
    if info.free < 30:
        return 2
    return 3


def app_queue_bucket(qsize: int) -> int:
    if qsize == 0:
        return 0
    if qsize < 5:
        return 1
    if qsize < 10:
        return 2
    return 3


def cca_fail_bucket(fails: int) -> int:
    if fails == 0:
        return 0
    if fails < 50:
        return 1
    if fails < 150:
        return 2
    return 3


def parse_neighbor_count(lines: List[str]) -> int:
    count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith(("| ext address", "ext address", "done")):
            continue
        if re.search(r"\b(child|router|reed)\b", lower) or re.search(r"\b0x[0-9a-f]+\b", lower):
            count += 1
    return count


def query_bufferinfo(ns: OTNS, node_id: int) -> BufferInfo:
    try:
        return parse_bufferinfo(ns.node_cmd(node_id, "bufferinfo"))
    except Exception as ex:
        return BufferInfo(raw=f"bufferinfo_error={ex}")



def parse_mac_counters(lines: List[str]) -> Dict[str, int]:
    counters = {}
    for line in lines:
        parts = line.strip().split(":", 1)
        if len(parts) == 2:
            key = parts[0].strip()
            try:
                counters[key] = int(parts[1].strip())
            except ValueError:
                pass
    return counters

def query_mac_counters(ns: OTNS, node_id: int) -> Dict[str, int]:
    try:
        return parse_mac_counters(ns.node_cmd(node_id, "counters mac"))
    except Exception:
        return {}

def query_neighbor_count(ns: OTNS, node_id: int) -> int:
    try:
        return parse_neighbor_count(ns.node_cmd(node_id, "neighbor table"))
    except Exception:
        return 0


def schedule_tx_in_window(reed: ReedRuntime, window_start: float, period: float) -> None:
    reed.tx_window_start = window_start
    reed.tx_period = period
    reed.tx_random_offset = reed.bandit.rng.uniform(0.0, period)
    reed.next_tx_time = window_start + reed.tx_random_offset


def schedule_next_tx_window(reed: ReedRuntime, sim_time: float, period: float) -> None:
    window_start = reed.tx_window_start + period
    while window_start <= sim_time + 1e-9:
        window_start += period
    schedule_tx_in_window(reed, window_start, period)


def collect_coap_counts(ns: OTNS, reeds: Dict[int, ReedRuntime], br_id: int) -> int:
    rx_at_br = 0
    for msg in ns.coaps() or []:
        if msg.get("uri") != "t":
            continue
        for receiver in msg.get("receivers", []) or []:
            dst = receiver.get("dst")
            if dst == br_id:
                rx_at_br += 1
                src = msg.get("src")
                if src in reeds:
                    reeds[src].rx_count += 1
    return rx_at_br


def set_radioparam(ns: OTNS, name: str, value: float) -> None:
    ns._do_command(f"radioparam {name} {value}")


def get_sim_time(ns: OTNS) -> float:
    return ns._expect_int(ns._do_command("time")) / 1.0e6


def run_otns_command(ns: OTNS, command: str) -> None:
    try:
        ns._do_command(command)
    except Exception:
        pass


def format_float_list(values: List[float]) -> str:
    return "|".join(f"{value:.6f}" for value in values)


def format_int_list(values: List[int]) -> str:
    return "|".join(str(value) for value in values)


def progress_log(log_file, message: str) -> None:
    if log_file is None:
        return
    log_file.write(f"{message}\n")
    log_file.flush()


def should_log_decision(decision_count: int, first_decisions: int, interval: int) -> bool:
    if decision_count <= first_decisions:
        return True
    return interval > 0 and decision_count % interval == 0


def write_qtable(output: str, reeds: Dict[int, ReedRuntime]) -> None:
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fieldnames = [
        "node_id",
        "node_name",
        "state",
        "neighbor_count",
        "app_queue_bucket",
        "queue_free_bucket",
        "cca_fail_bucket",
        "action_index",
        "action_adjustment_s",
        "q_value",
        "action_count",
    ]

    with open(output, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for reed in sorted(reeds.values(), key=lambda item: item.node_id):
            for state in sorted(reed.bandit.q_values):
                q_values = reed.bandit.q_values[state]
                counts = reed.bandit.counts[state]
                for action_index, action in enumerate(reed.bandit.actions):
                    writer.writerow({
                        "node_id": reed.node_id,
                        "node_name": reed.name,
                        "state": f"{state[0]}:{state[1]}:{state[2]}:{state[3]}",
                        "neighbor_count": state[0],
                        "app_queue_bucket": state[1],
                        "queue_free_bucket": state[2],
                        "cca_fail_bucket": state[3],
                        "action_index": action_index,
                        "action_adjustment_s": f"{action:.1f}",
                        "q_value": f"{q_values[action_index]:.6f}",
                        "action_count": counts[action_index],
                    })


def build_topology(ns: OTNS) -> Tuple[int, Dict[int, ReedRuntime]]:
    ns.radiomodel = "MutualInterference"
    set_radioparam(ns, "MeterPerUnit", METER_PER_UNIT)
    ns.set_title("BR + 5x5 REED contextual bandit RL")
    ns.config_visualization(broadcast_message=False, router_table=True)

    br = ns.add("br", x=BR_POS[0], y=BR_POS[1], radio_range=RADIO_RANGE_M)
    ns.node_cmd(br, "coap start")
    ns.node_cmd(br, "coap resource t")

    rng = random.Random(RANDOM_SEED)
    reeds: Dict[int, ReedRuntime] = {}
    for row in range(5):
        for col in range(5):
            x = col * GRID_SPACING_M
            y = (row + 1) * GRID_SPACING_M
            node_id = ns.add("reed", x=x, y=y, radio_range=RADIO_RANGE_M)
            ns.node_cmd(node_id, "routerselectionjitter 1")
            ns.node_cmd(node_id, "coap start")
            reeds[node_id] = ReedRuntime(
                node_id=node_id,
                name=f"reed{row * 5 + col + 1}",
                bandit=ContextualEpsilonGreedyBandit(REED_ACTIONS, EPSILON, rng),
            )

    return br, reeds


def open_web(ns: OTNS) -> None:
    try:
        ns.web("main")
    except TypeError:
        ns.web()


def run(output: str,
        web: bool,
        speed: float,
        bandit_log_output: str,
        qtable_output: str,
        progress_log_output: str,
        progress_log_first_decisions: int,
        progress_log_interval: int) -> None:
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(bandit_log_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(progress_log_output) or ".", exist_ok=True)

    progress_file = open(progress_log_output, "w")
    ns = OTNS(otns_args=["-log", "info", "-watch", "warn"])
    ns.speed = speed
    if web:
        open_web(ns)

    try:
        progress_log(progress_file, "topology build start")
        br, reeds = build_topology(ns)
        progress_log(progress_file, f"topology build done: br={br}, reeds={','.join(str(item) for item in sorted(reeds))}")
        progress_log(progress_file, f"formation go start: duration={FORMATION_TIME:.6f}s")
        ns.go(FORMATION_TIME)
        progress_log(progress_file, f"formation go done: sim_time={get_sim_time(ns):.6f}s")
        progress_log(progress_file, "coaps enable start")
        ns.coaps_enable()
        progress_log(progress_file, "coaps enable done")

        sim_time = get_sim_time(ns)
        next_decision_time = sim_time + DECISION_INTERVAL
        
        for reed in reeds.values():
            reed.last_app_gen_time = sim_time
            schedule_tx_in_window(reed, sim_time, INITIAL_TX_PERIOD)
            mac_counters = query_mac_counters(ns, reed.node_id)
            reed.last_tx_err_cca = mac_counters.get("TxErrCca", 0)

            before_info = query_bufferinfo(ns, reed.node_id)
            neighbor_count = query_neighbor_count(ns, reed.node_id)
            state = (neighbor_count, app_queue_bucket(reed.app_queue), free_queue_bucket(before_info), cca_fail_bucket(0))
            action, action_index, explored = reed.bandit.select(state)
            
            reed.last_state = state
            reed.last_action_index = action_index
            reed.tx_period = max(0.1, min(10.0, INITIAL_TX_PERIOD * action))
            
            progress_log(
                progress_file,
                f"initial schedule: node={reed.node_id} window_start={reed.tx_window_start:.6f} "
                f"period={reed.tx_period:.6f} offset={reed.tx_random_offset:.6f} next={reed.next_tx_time:.6f}",
            )

        fieldnames = [
            "sim_time_s", "node_id", "node_name", "neighbor_count", "app_queue", "app_queue_bucket", "queue_free", "queue_total",
            "queue_free_bucket", "cca_fail_bucket", "state", "action_adjustment_s", "new_period_s", "action_index", "explored",
            "app_drops_interval", "cca_fails_interval", "penalty", "reward", "node_coap_tx_count", "node_coap_rx_at_br_count", "br_total_rx_count"
        ]

        br_total_rx_count = 0
        decision_count = 0
        with open(output, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            csvfile.flush()
            progress_log(progress_file, "csv headers written")

            while sim_time < SIMULATION_TIME:
                next_tx = min(reed.next_tx_time for reed in reeds.values())
                next_app_gen = min(reed.last_app_gen_time + APP_MESSAGE_PERIOD for reed in reeds.values())
                next_event_time = min(next_tx, next_decision_time, next_app_gen)

                if next_event_time > SIMULATION_TIME:
                    remaining = SIMULATION_TIME - sim_time
                    if remaining > TIME_EPSILON_S:
                        ns.go(remaining)
                    break

                time_diff = next_event_time - sim_time
                if time_diff > 1e-6:
                    ns.go(max(time_diff, MIN_GO_DURATION_S))
                    sim_time = get_sim_time(ns)
                else:
                    sim_time = get_sim_time(ns)

                br_total_rx_count += collect_coap_counts(ns, reeds, br)

                # Process Message Generation
                for reed in reeds.values():
                    while sim_time >= reed.last_app_gen_time + APP_MESSAGE_PERIOD - TIME_EPSILON_S:
                        if reed.app_queue < MAX_APP_QUEUE_SIZE:
                            reed.app_queue += 1
                        else:
                            reed.app_drop_count += 1
                        reed.last_app_gen_time += APP_MESSAGE_PERIOD

                # Process Transmissions
                due_reeds = [r for r in reeds.values() if r.next_tx_time <= sim_time + TIME_EPSILON_S]
                if not due_reeds and sim_time < next_decision_time - TIME_EPSILON_S and sim_time < next_app_gen - TIME_EPSILON_S:
                    ns.go(MIN_GO_DURATION_S)
                    sim_time = get_sim_time(ns)
                    
                    for reed in reeds.values():
                        while sim_time >= reed.last_app_gen_time + APP_MESSAGE_PERIOD - TIME_EPSILON_S:
                            if reed.app_queue < MAX_APP_QUEUE_SIZE:
                                reed.app_queue += 1
                            else:
                                reed.app_drop_count += 1
                            reed.last_app_gen_time += APP_MESSAGE_PERIOD
                            
                    due_reeds = [r for r in reeds.values() if r.next_tx_time <= sim_time + TIME_EPSILON_S]
                    
                for reed in due_reeds:
                    # Send one message per transmission window if queue has packets
                    if reed.app_queue > 0:
                        run_otns_command(ns, f"send coap {reed.node_id} {br} datasize {COAP_PAYLOAD_BYTES}")
                        reed.app_queue -= 1
                        reed.tx_count += 1
                    
                    schedule_next_tx_window(reed, sim_time, reed.tx_period)

                # Process Decision Intervals
                if sim_time >= next_decision_time - TIME_EPSILON_S:
                    decision_count += 1
                    for reed in reeds.values():
                        mac_counters = query_mac_counters(ns, reed.node_id)
                        current_tx_err_cca = mac_counters.get("TxErrCca", 0)
                        cca_fails = current_tx_err_cca - reed.last_tx_err_cca
                        reed.last_tx_err_cca = current_tx_err_cca
                        
                        app_drops = reed.app_drop_count - reed.last_app_drop_count
                        reed.last_app_drop_count = reed.app_drop_count

                        after_info = query_bufferinfo(ns, reed.node_id)
                        
                        # Reward: Positive reward for maintaining queue, heavily weighted against CCA fails and Drops
                        efficiency_reward = 0.1 if after_info.normalized_free >= 0.2 else 0.0
                        queue_penalty = app_drops * 10.0
                        fail_penalty = cca_fails * 0.05
                        penalty = queue_penalty + fail_penalty
                        reward = efficiency_reward - penalty
                        
                        if reed.last_state is not None and reed.last_action_index is not None:
                            reed.bandit.update(reed.last_state, reed.last_action_index, reward)
                            
                        neighbor_count = query_neighbor_count(ns, reed.node_id)
                        state = (neighbor_count, app_queue_bucket(reed.app_queue), free_queue_bucket(after_info), cca_fail_bucket(cca_fails))
                        action, action_index, explored = reed.bandit.select(state)
                        
                        new_period = max(0.1, min(10.0, reed.tx_period * action))
                        reed.tx_period = new_period
                        reed.last_state = state
                        reed.last_action_index = action_index
                        
                        writer.writerow({
                            "sim_time_s": f"{sim_time:.6f}",
                            "node_id": reed.node_id,
                            "node_name": reed.name,
                            "neighbor_count": neighbor_count,
                            "app_queue": reed.app_queue,
                            "app_queue_bucket": state[1],
                            "queue_free": "" if after_info.free is None else after_info.free,
                            "queue_total": "" if after_info.total is None else after_info.total,
                            "queue_free_bucket": state[2],
                            "cca_fail_bucket": state[3],
                            "state": f"{state[0]}:{state[1]}:{state[2]}:{state[3]}",
                            "action_adjustment_s": f"{action:.1f}",
                            "new_period_s": f"{new_period:.3f}",
                            "action_index": action_index,
                            "explored": int(explored),
                            "app_drops_interval": app_drops,
                            "cca_fails_interval": cca_fails,
                            "penalty": f"{penalty:.6f}",
                            "reward": f"{reward:.6f}",
                            "node_coap_tx_count": reed.tx_count,
                            "node_coap_rx_at_br_count": reed.rx_count,
                            "br_total_rx_count": br_total_rx_count,
                        })

                    csvfile.flush()
                    next_decision_time += DECISION_INTERVAL

            ns.go(2.0)
            br_total_rx_count += collect_coap_counts(ns, reeds, br)
        write_qtable(qtable_output, reeds)

    finally:
        progress_log(progress_file, "closing OTNS")
        ns.close()
        progress_log(progress_file, "closed OTNS")
        progress_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BR + 5x5 REED contextual bandit RL CoAP traffic simulation.")
    parser.add_argument("--output", default="tmp/br_reed_bandit_rl.csv", help="CSV output path.")
    parser.add_argument("--bandit-log-output",
                        default="tmp/br_reed_bandit_rl_bandit_decisions.csv",
                        help="Bandit decision log CSV output path.")
    parser.add_argument("--qtable-output",
                        default="tmp/br_reed_bandit_rl_qtable.csv",
                        help="Final per-node contextual bandit table CSV output path.")
    parser.add_argument("--progress-log-output",
                        default="tmp/br_reed_bandit_rl_progress.log",
                        help="Progress log path for locating long-running OTNS commands.")
    parser.add_argument("--progress-log-first-decisions",
                        type=int,
                        default=30,
                        help="Number of initial bandit decisions to log in detail.")
    parser.add_argument("--progress-log-interval",
                        type=int,
                        default=500,
                        help="After the initial decisions, log one detailed decision every N decisions. Use 0 to disable.")
    parser.add_argument("--no-web", action="store_true", help="Do not open the OTNS web visualization.")
    parser.add_argument("--speed", type=float, default=OTNS.MAX_SIMULATE_SPEED, help="OTNS simulation speed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(output=args.output,
        web=not args.no_web,
        speed=args.speed,
        bandit_log_output=args.bandit_log_output,
        qtable_output=args.qtable_output,
        progress_log_output=args.progress_log_output,
        progress_log_first_decisions=args.progress_log_first_decisions,
        progress_log_interval=args.progress_log_interval)


if __name__ == "__main__":
    try:
        main()
    except OTNSExitedError as ex:
        if ex.exit_code != 0:
            raise
