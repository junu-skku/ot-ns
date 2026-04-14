#!/usr/bin/env python3
# Copyright (c) 2026, The OTNS Authors.
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

"""
MLE Trickle Timer Vicious Cycle Experiment (standalone)
=======================================================

Demonstrates the Wi-Fi interference -> Trickle reset -> MLE Advertisement
explosion -> channel congestion -> more resets feedback loop in a 30-REED /
1-BR Thread network.

Phases:
  1. Topology build   - sequential BR + 30 REED placement with settle gaps
  2. Stabilization    - wait for Trickle timers to reach maximum (low MLE adv)
  3. CoAP baseline    - periodic CoAP to BR; confirm MLE stays low
  4. Wi-Fi burst #1   - moderate interference, observe initial spike
  5. Wi-Fi burst #2   - stronger interference, observe amplified spike
  6. Sustained storm  - continuous high interference, observe vicious cycle
  7. Recovery         - interference off, observe whether network calms down

All per-interval metrics are written to CSV for offline analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_PYLIBS_ROOT = Path(__file__).resolve().parents[1]
if str(_PYLIBS_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYLIBS_ROOT))

from otns.cli import OTNS
from otns.cli.errors import OTNSCliError, OTNSExitedError

DEFAULT_OUTPUT_DIR = "tmp/mle_trickle_vicious_cycle"
NODE_INIT_SETTLE_SECONDS = 1.0

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_metric_name(text: str) -> str:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text.strip())
    text = re.sub(r"[\s\-/]+", "_", text)
    text = re.sub(r"[^0-9A-Za-z_]", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def parse_counter_lines(lines: Sequence[str]) -> Dict[str, int]:
    counters: Dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if ":" in stripped:
            key_text, value_text = stripped.split(":", 1)
            key = normalize_metric_name(key_text)
            value_text = value_text.strip()
            if key and re.fullmatch(r"-?\d+", value_text):
                counters[key] = int(value_text)
                continue
        parts = stripped.split()
        if len(parts) >= 2 and re.fullmatch(r"-?\d+", parts[-1]):
            key = normalize_metric_name(" ".join(parts[:-1]))
            if key:
                counters[key] = int(parts[-1])
    return counters


def percentile(values: Sequence[int], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * ratio
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    frac = rank - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, sort_keys=True)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# OTNS CLI wrappers (safe / compat)
# ---------------------------------------------------------------------------

def try_node_cmd(ns: OTNS, node_id: int, command: str) -> Tuple[bool, List[str]]:
    try:
        if hasattr(ns, "node_cmd"):
            return True, ns.node_cmd(node_id, command)
        if hasattr(ns, "cmd"):
            return True, ns.cmd(f'node {node_id} "{command}"')
    except (OTNSCliError, AttributeError):
        return False, []
    return False, []


def safe_node_cmd(ns: OTNS, node_id: int, command: str) -> List[str]:
    ok, output = try_node_cmd(ns, node_id, command)
    return output if ok else []


def safe_node_cmd_retry(ns: OTNS, node_id: int, command: str,
                        retries: int = 3, settle_seconds: float = 0.2) -> List[str]:
    output: List[str] = []
    for _ in range(max(1, retries)):
        ok, output = try_node_cmd(ns, node_id, command)
        if ok:
            return output
        safe_go(ns, settle_seconds)
    return output


def try_cmd(ns: OTNS, command: str) -> Tuple[bool, List[str]]:
    try:
        if hasattr(ns, "cmd"):
            return True, ns.cmd(command)
        if hasattr(ns, "_do_command"):
            return True, ns._do_command(command)
    except (OTNSCliError, AttributeError):
        return False, []
    return False, []


def safe_cmd(ns: OTNS, command: str) -> List[str]:
    ok, output = try_cmd(ns, command)
    return output if ok else []


def safe_go(ns: OTNS, duration: float) -> List[str]:
    try:
        return ns.go(duration)
    except (OTNSCliError, AttributeError):
        return []


def compat_set_radioparam(ns: OTNS, name: str, value: float) -> None:
    if hasattr(ns, "set_radioparam"):
        try:
            ns.set_radioparam(name, value)
            return
        except Exception:
            pass
    safe_cmd(ns, f"radioparam {name} {value}")


def compat_watch_default(ns: OTNS, level: str) -> None:
    if hasattr(ns, "watch_default"):
        try:
            ns.watch_default(level)
            return
        except Exception:
            pass
    safe_cmd(ns, f"watch default {level}")


def compat_watch_all(ns: OTNS, level: str) -> None:
    if hasattr(ns, "watch_all"):
        try:
            ns.watch_all(level)
            return
        except Exception:
            pass
    safe_cmd(ns, f"watch all {level}")


def get_state(ns: OTNS, node_id: int) -> str:
    lines = safe_node_cmd(ns, node_id, "state")
    return lines[0].strip().lower() if lines else "unknown"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NodeInfo:
    node_id: int
    role: str
    cluster: str
    x: int
    y: int


@dataclass
class NodeBundle:
    br_id: int
    routers: List[NodeInfo]
    candidates: List[NodeInfo]
    meds: List[NodeInfo]
    seds: List[NodeInfo]
    wifi_nodes: List[NodeInfo]

    @property
    def thread_nodes(self) -> List[NodeInfo]:
        return [NodeInfo(self.br_id, "br", "br", 0, 0)] + self.routers + self.candidates + self.meds + self.seds

    @property
    def all_thread_node_ids(self) -> List[int]:
        return [n.node_id for n in self.thread_nodes]

    @property
    def candidate_ids(self) -> List[int]:
        return [n.node_id for n in self.candidates]

    @property
    def router_like_ids(self) -> List[int]:
        return [self.br_id] + [n.node_id for n in self.routers + self.candidates]

    @property
    def router_capable_ids(self) -> List[int]:
        return [n.node_id for n in self.routers + self.candidates]

    @property
    def end_device_ids(self) -> List[int]:
        return [n.node_id for n in self.meds + self.seds]


@dataclass
class CounterSample:
    time_s: int
    active_router_count: int
    state_changes: int
    tx_total: int
    tx_retry: int
    tx_err_cca: int
    tx_err_busy_channel: int
    rx_err_fcs: int
    attach_attempts: int
    better_parent_attach_attempts: int
    parent_changes: int
    role_entry_events: int
    coap_sent: int
    coap_delivered: int
    coap_delivery_ratio: float
    coap_latency_ms_mean: float
    coap_latency_ms_p95: float
    partitions: int
    tx_time: int
    rx_time: int
    sleep_time: int
    deep_mle_advertisement: int = 0
    deep_mle_link_request: int = 0
    deep_mle_link_accept: int = 0
    deep_address_solicit: int = 0
    deep_address_notify: int = 0
    deep_parent_request: int = 0
    deep_parent_response: int = 0

    @property
    def cca_fail_ratio(self) -> float:
        return safe_ratio(self.tx_err_cca, self.tx_total)

    @property
    def busy_channel_ratio(self) -> float:
        return safe_ratio(self.tx_err_busy_channel, self.tx_total)

    @property
    def retry_ratio(self) -> float:
        return safe_ratio(self.tx_retry, self.tx_total)

    @property
    def role_churn(self) -> int:
        return self.state_changes + self.parent_changes + self.attach_attempts


@dataclass
class CoapWindowStats:
    sent_commands: int = 0
    delivered_requests: int = 0
    latencies_us: List[int] = field(default_factory=list)

    def ingest(self, messages: Sequence[Dict[str, object]], tracked_sources: Iterable[int]) -> None:
        tracked = set(tracked_sources)
        for msg in messages:
            src = as_int(msg.get("src"), -1)
            code = as_int(msg.get("code"), 0)
            if src not in tracked or code >= 64:
                continue
            receivers = msg.get("receivers") or []
            if receivers:
                self.delivered_requests += 1
            sent_time = as_int(msg.get("time"), 0)
            for receiver in receivers:
                if not isinstance(receiver, dict):
                    continue
                recv_time = as_int(receiver.get("time"), sent_time)
                self.latencies_us.append(max(0, recv_time - sent_time))

    def reset(self) -> None:
        self.sent_commands = 0
        self.delivered_requests = 0
        self.latencies_us.clear()


@dataclass
class DeepLogCollector:
    enabled: bool
    interval_counts: Dict[str, int] = field(default_factory=dict)
    total_counts: Dict[str, int] = field(default_factory=dict)
    patterns: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: {
        "mle_advertisement": ("mle", "advert"),
        "mle_link_request": ("mle", "link request"),
        "mle_link_accept": ("mle", "link accept"),
        "address_solicit": ("address solicit",),
        "address_notify": ("address notify",),
        "parent_request": ("mle", "parent request"),
        "parent_response": ("mle", "parent response"),
    })
    _fps: Dict[int, object] = field(default_factory=dict)

    def drain_node_logs(self, node_ids: Sequence[int]) -> List[str]:
        lines = []
        for nid in node_ids:
            if nid not in self._fps:
                path = Path(f"tmp/0_{nid}.log")
                if path.exists():
                    self._fps[nid] = path.open("r", encoding="utf-8", errors="ignore")
            if nid in self._fps:
                lines.extend(self._fps[nid].readlines())
        return lines

    def feed(self, lines: Sequence[str], event_timeline: List[Dict[str, object]],
             run_label: str, time_s: int) -> None:
        if not self.enabled:
            return
        for raw in lines:
            lower = raw.lower()
            for key, tokens in self.patterns.items():
                if all(token in lower for token in tokens):
                    self.interval_counts[key] = self.interval_counts.get(key, 0) + 1
                    self.total_counts[key] = self.total_counts.get(key, 0) + 1
                    break

    def drain_interval(self) -> Dict[str, int]:
        counts = dict(self.interval_counts)
        self.interval_counts.clear()
        return counts


# ---------------------------------------------------------------------------
# Topology / node helpers
# ---------------------------------------------------------------------------

def paper_grid_positions(count: int, spacing: int, origin_x: int, origin_y: int) -> List[Tuple[int, int]]:
    cols = max(1, int(math.ceil(math.sqrt(count))))
    positions: List[Tuple[int, int]] = []
    for idx in range(count):
        row = idx // cols
        col = idx % cols
        positions.append((origin_x + col * spacing, origin_y + row * spacing))
    return positions


def add_wifi_nodes(ns: OTNS, count: int, center_x: int, center_y: int,
                   spacing: int) -> List[NodeInfo]:
    nodes: List[NodeInfo] = []
    for idx in range(count):
        x = center_x + (idx - (count // 2)) * spacing
        y = center_y
        node_id = ns.add("wifi", x=x, y=y, radio_range=90)
        safe_cmd(ns, f"rfsim {node_id} txintf 0")
        safe_go(ns, NODE_INIT_SETTLE_SECONDS)
        nodes.append(NodeInfo(node_id, "wifi", "wifi", x, y))
    return nodes


def set_wifi_interference(ns: OTNS, wifi_nodes: Sequence[NodeInfo], txintf: int) -> None:
    for node in wifi_nodes:
        safe_cmd(ns, f"rfsim {node.node_id} txintf {txintf}")


def build_candidate_init_script(upgrade_threshold: int, downgrade_threshold: int,
                                routerselectionjitter: int) -> str:
    return "\n".join([
        "dataset init new",
        "dataset networkname otns",
        "dataset networkkey 00112233445566778899aabbccddeeff",
        "dataset panid 0xface",
        "dataset channel 11",
        "dataset extpanid dead00beef00cafe",
        "dataset meshlocalprefix fdde:ad00:beef:0::",
        "dataset pskc 3aa55f91ca47d1e4e71a08cb35e91591",
        "dataset commit active",
        f"routerselectionjitter {routerselectionjitter}",
        f"routerupgradethreshold {upgrade_threshold}",
        f"routerdowngradethreshold {downgrade_threshold}",
        "routereligible disable",
        "ifconfig up",
        "thread start",
    ])


def enable_candidate_router_eligibility(ns: OTNS, candidate_ids: Sequence[int],
                                        settle_seconds: float = NODE_INIT_SETTLE_SECONDS) -> None:
    for node_id in candidate_ids:
        safe_node_cmd_retry(ns, node_id, "routereligible enable", settle_seconds=settle_seconds)
        safe_go(ns, settle_seconds)


def reset_observation_window(ns: OTNS, bundle: NodeBundle) -> None:
    for node_id in bundle.all_thread_node_ids:
        safe_node_cmd(ns, node_id, "counters mac reset")
        safe_node_cmd(ns, node_id, "counters mle reset")


# ---------------------------------------------------------------------------
# Network readiness
# ---------------------------------------------------------------------------

def wait_for_network_readiness(ns: OTNS, bundle: NodeBundle,
                               max_wait_seconds: int, stable_seconds: int,
                               event_timeline: List[Dict[str, object]],
                               run_label: str, start_time_s: int) -> int:
    if max_wait_seconds <= 0 or stable_seconds <= 0:
        return 0
    consecutive_ready = 0
    waited_seconds = 0
    while waited_seconds < max_wait_seconds:
        states = {nid: get_state(ns, nid) for nid in bundle.all_thread_node_ids}
        connected_rl = sum(1 for nid in bundle.router_like_ids
                           if states.get(nid, "unknown") in ("leader", "router", "child"))
        connected_ed = sum(1 for nid in bundle.end_device_ids
                           if states.get(nid, "unknown") == "child")
        try:
            partition_count = len(ns.partitions())
        except OTNSCliError:
            partition_count = 0
        if (connected_rl == len(bundle.router_like_ids)
                and connected_ed == len(bundle.end_device_ids)
                and partition_count <= 1):
            consecutive_ready += 1
            if consecutive_ready >= stable_seconds:
                event_timeline.append({
                    "run": run_label, "time_s": start_time_s + waited_seconds,
                    "type": "network_ready", "waited_seconds": waited_seconds,
                })
                return waited_seconds
        else:
            consecutive_ready = 0
        safe_go(ns, 1)
        waited_seconds += 1
    event_timeline.append({
        "run": run_label, "time_s": start_time_s + waited_seconds,
        "type": "network_ready_timeout", "waited_seconds": waited_seconds,
    })
    return waited_seconds


# ---------------------------------------------------------------------------
# Counter collection / interval sample
# ---------------------------------------------------------------------------

def diff_counter_maps(previous: Dict[int, Dict[str, int]],
                      current: Dict[int, Dict[str, int]]) -> Dict[int, Dict[str, int]]:
    delta: Dict[int, Dict[str, int]] = {}
    for node_id, cur_map in current.items():
        prev_map = previous.get(node_id, {})
        row: Dict[str, int] = {}
        for key, value in cur_map.items():
            old = prev_map.get(key, 0)
            row[key] = value - old if value >= old else value
        delta[node_id] = row
    return delta


def sum_keys(counter_maps: Dict[int, Dict[str, int]], exact: Sequence[str],
             contains: Sequence[str] = ()) -> int:
    total = 0
    for counters in counter_maps.values():
        matched_exact = False
        for key in exact:
            if key in counters:
                total += counters[key]
                matched_exact = True
        if matched_exact or not contains:
            continue
        for key, value in counters.items():
            if any(token in key for token in contains):
                total += value
    return total


def collect_counter_snapshot(ns: OTNS, bundle: NodeBundle) -> Tuple[
    Dict[int, str], Dict[int, Dict[str, int]],
    Dict[int, Dict[str, int]], Dict[int, Dict[str, int]],
]:
    states: Dict[int, str] = {}
    mac: Dict[int, Dict[str, int]] = {}
    mle: Dict[int, Dict[str, int]] = {}
    radio: Dict[int, Dict[str, int]] = {}
    for node_id in bundle.all_thread_node_ids:
        states[node_id] = get_state(ns, node_id)
        mac[node_id] = parse_counter_lines(safe_node_cmd(ns, node_id, "counters mac"))
        mle[node_id] = parse_counter_lines(safe_node_cmd(ns, node_id, "counters mle"))
        radio[node_id] = {}
    return states, mac, mle, radio


def make_interval_sample(
    ns: OTNS, bundle: NodeBundle,
    current_states: Dict[int, str], previous_states: Dict[int, str],
    current_mac: Dict[int, Dict[str, int]], previous_mac: Dict[int, Dict[str, int]],
    current_mle: Dict[int, Dict[str, int]], previous_mle: Dict[int, Dict[str, int]],
    current_radio: Dict[int, Dict[str, int]], previous_radio: Dict[int, Dict[str, int]],
    time_s: int, coap_window: CoapWindowStats, deep_counts: Dict[str, int],
) -> CounterSample:
    mac_delta = diff_counter_maps(previous_mac, current_mac)
    mle_delta = diff_counter_maps(previous_mle, current_mle)
    radio_delta = diff_counter_maps(previous_radio, current_radio)

    active_router_count = sum(
        1 for nid in bundle.router_like_ids
        if current_states.get(nid) in ("leader", "router")
    )
    state_changes = sum(
        1 for nid in bundle.all_thread_node_ids
        if previous_states.get(nid) is not None and previous_states.get(nid) != current_states.get(nid)
    )

    lat_mean_ms = (mean(coap_window.latencies_us) / 1000.0) if coap_window.latencies_us else 0.0
    lat_p95_ms = (percentile(coap_window.latencies_us, 0.95) / 1000.0) if coap_window.latencies_us else 0.0

    try:
        partitions = len(ns.partitions())
    except OTNSCliError:
        partitions = 0

    return CounterSample(
        time_s=time_s,
        active_router_count=active_router_count,
        state_changes=state_changes,
        tx_total=sum_keys(mac_delta, ["tx_total"], ["tx_total"]),
        tx_retry=sum_keys(mac_delta, ["tx_retry"], ["retry"]),
        tx_err_cca=sum_keys(mac_delta, ["tx_err_cca"], ["cca"]),
        tx_err_busy_channel=sum_keys(mac_delta, ["tx_err_busy_channel"], ["busy_channel"]),
        rx_err_fcs=sum_keys(mac_delta, ["rx_err_fcs"], ["fcs"]),
        attach_attempts=sum_keys(mle_delta, ["attach_attempts", "attach_attempt"], ["attach_attempt"]),
        better_parent_attach_attempts=sum_keys(mle_delta, ["better_parent_attach_attempts", "better_parent_attach_attempt"], ["better_parent_attach"]),
        parent_changes=sum_keys(mle_delta, ["parent_changes", "parent_change"], ["parent_change"]),
        role_entry_events=sum_keys(mle_delta, [], ["role", "become"]),
        coap_sent=coap_window.sent_commands,
        coap_delivered=coap_window.delivered_requests,
        coap_delivery_ratio=safe_ratio(coap_window.delivered_requests, coap_window.sent_commands),
        coap_latency_ms_mean=round(lat_mean_ms, 3),
        coap_latency_ms_p95=round(lat_p95_ms, 3),
        partitions=partitions,
        tx_time=sum_keys(radio_delta, ["tx_time"], ["tx_time"]),
        rx_time=sum_keys(radio_delta, ["rx_time"], ["rx_time"]),
        sleep_time=sum_keys(radio_delta, ["sleep_time"], ["sleep_time"]),
        deep_mle_advertisement=deep_counts.get("mle_advertisement", 0),
        deep_mle_link_request=deep_counts.get("mle_link_request", 0),
        deep_mle_link_accept=deep_counts.get("mle_link_accept", 0),
        deep_address_solicit=deep_counts.get("address_solicit", 0),
        deep_address_notify=deep_counts.get("address_notify", 0),
        deep_parent_request=deep_counts.get("parent_request", 0),
        deep_parent_response=deep_counts.get("parent_response", 0),
    )


# ---------------------------------------------------------------------------
# CoAP helpers
# ---------------------------------------------------------------------------

def collect_coap_records(ns: OTNS, run_label: str, time_s: int, fp,
                         coap_window: CoapWindowStats,
                         tracked_sources: Sequence[int]) -> None:
    try:
        messages = ns.coaps() or []
    except OTNSCliError:
        messages = []
    coap_window.ingest(messages, tracked_sources)
    for message in messages:
        fp.write(json.dumps({"run": run_label, "time_s": time_s, "message": message}, sort_keys=True) + "\n")


def get_coap_sender_ids(bundle: NodeBundle, sender_class: str) -> List[int]:
    if sender_class == "router_capable":
        return bundle.router_capable_ids
    if sender_class == "end_devices":
        return bundle.end_device_ids
    return bundle.all_thread_node_ids


def select_coap_senders(node_ids: Sequence[int], args: argparse.Namespace, sim_time_s: int) -> List[int]:
    if args.coap_period_seconds <= 0 or sim_time_s % args.coap_period_seconds != 0:
        return []
    if not node_ids:
        return []
    if args.senders_per_tick <= 0 or args.senders_per_tick >= len(node_ids):
        return list(node_ids)
    tick_index = sim_time_s // args.coap_period_seconds
    start = (tick_index * args.senders_per_tick) % len(node_ids)
    return [node_ids[(start + offset) % len(node_ids)] for offset in range(args.senders_per_tick)]


def send_periodic_coap(ns: OTNS, sender_ids: Sequence[int], br_id: int, payload_size: int) -> int:
    sent = 0
    for node_id in sender_ids:
        try:
            ok, _ = try_cmd(ns, f"send coap con {node_id} {br_id} rloc datasize {payload_size}")
            if ok:
                sent += 1
        except OTNSCliError:
            continue
    return sent


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

def copy_run_artifacts(run_dir: Path) -> None:
    for src_name, dst_name in (
        ("tmp/0_stats.csv", "stats.csv"),
        ("tmp/0_txbytes.csv", "txbytes.csv"),
        ("tmp/0_chansamples.csv", "chansamples.csv"),
        ("current.pcap", "current.pcap"),
    ):
        src = Path(src_name)
        if src.exists():
            shutil.copy(src, run_dir / dst_name)


def collect_final_node_metrics(ns: OTNS, bundle: NodeBundle, run_label: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    node_lookup: Dict[int, NodeInfo] = {
        n.node_id: n for n in bundle.routers + bundle.candidates + bundle.meds + bundle.seds + bundle.wifi_nodes
    }
    node_lookup[bundle.br_id] = NodeInfo(bundle.br_id, "br", "br", 0, 0)
    for node_id, node in sorted(node_lookup.items()):
        rows.append({
            "run": run_label,
            "node_id": node_id,
            "declared_role": node.role,
            "cluster": node.cluster,
            "x": node.x, "y": node.y,
            "state": get_state(ns, node_id),
            "mac_counters": parse_counter_lines(safe_node_cmd(ns, node_id, "counters mac")),
            "mle_counters": parse_counter_lines(safe_node_cmd(ns, node_id, "counters mle")),
        })
    return rows


# ---------------------------------------------------------------------------
# Topology builder
# ---------------------------------------------------------------------------

def build_reed_grid(ns: OTNS, args: argparse.Namespace) -> NodeBundle:
    compat_set_radioparam(ns, "MeterPerUnit", 1.0)

    positions = paper_grid_positions(
        args.reed_count, args.grid_spacing,
        args.grid_origin_x, args.grid_origin_y,
    )
    min_x = min(p[0] for p in positions)
    max_x = max(p[0] for p in positions)
    min_y = min(p[1] for p in positions)
    max_y = max(p[1] for p in positions)

    br_x = int(round((min_x + max_x) / 2.0))
    br_y = max(5, min_y - args.br_offset_y)

    br_id = ns.add("br", x=br_x, y=br_y,
                    radio_range=args.thread_range, txpower=args.thread_txpower)
    safe_go(ns, 10)
    print(f"  BR node {br_id} at ({br_x}, {br_y})")

    candidate_script = build_candidate_init_script(
        args.router_upgrade_threshold,
        args.router_downgrade_threshold,
        args.router_selection_jitter,
    )

    candidates: List[NodeInfo] = []
    for i, (x, y) in enumerate(positions):
        node_id = ns.add("router", x=x, y=y,
                         radio_range=args.thread_range, txpower=args.thread_txpower,
                         script=candidate_script)
        candidates.append(NodeInfo(node_id, "router", "reed_grid", x, y))
        if (i + 1) % 5 == 0:
            safe_go(ns, 3)
    safe_go(ns, 30)
    print(f"  {len(candidates)} REED candidates placed")

    wifi_cx = int(round((min_x + max_x) / 2.0 + args.grid_spacing * 0.75))
    wifi_cy = int(round((min_y + max_y) / 2.0))
    wifi_nodes = add_wifi_nodes(ns, args.wifi_node_count, wifi_cx, wifi_cy, args.wifi_spacing)
    safe_go(ns, 5)
    print(f"  {len(wifi_nodes)} Wi-Fi interferer(s) placed (inactive)")

    return NodeBundle(br_id=br_id, routers=[], candidates=candidates,
                      meds=[], seds=[], wifi_nodes=wifi_nodes)


# ---------------------------------------------------------------------------
# Phase logic
# ---------------------------------------------------------------------------

def is_stable(sample: CounterSample, args: argparse.Namespace) -> bool:
    return (
        sample.partitions <= 1
        and sample.deep_mle_advertisement <= args.stable_mle_threshold
        and sample.role_churn <= args.stable_churn_threshold
        and sample.parent_changes <= args.stable_parent_threshold
    )


def infer_trickle_reset(sample: CounterSample, baseline_mle: float,
                        baseline_cca: float, baseline_retry: float,
                        args: argparse.Namespace) -> bool:
    if baseline_mle <= 0:
        return False
    mle_burst = sample.deep_mle_advertisement >= max(
        args.reset_min_advertisements,
        baseline_mle * args.reset_factor,
    )
    cca_rise = sample.cca_fail_ratio >= baseline_cca + args.reset_cca_delta
    retry_rise = sample.retry_ratio >= baseline_retry + args.reset_retry_delta
    return mle_burst and (cca_rise or retry_rise)


def make_row(phase: str, cycle: int, wifi_txintf: int,
             sample: CounterSample, baseline_mle: float,
             trickle_reset: bool) -> Dict[str, object]:
    mle_ratio = safe_ratio(sample.deep_mle_advertisement, max(1e-9, baseline_mle)) if baseline_mle > 0 else 0.0
    return {
        "time_s": sample.time_s,
        "phase": phase,
        "cycle": cycle,
        "wifi_txintf": wifi_txintf,
        "active_router_count": sample.active_router_count,
        "tx_total": sample.tx_total,
        "tx_retry": sample.tx_retry,
        "tx_err_cca": sample.tx_err_cca,
        "tx_err_busy_channel": sample.tx_err_busy_channel,
        "rx_err_fcs": sample.rx_err_fcs,
        "cca_fail_ratio": round(sample.cca_fail_ratio, 6),
        "retry_ratio": round(sample.retry_ratio, 6),
        "role_churn": sample.role_churn,
        "attach_attempts": sample.attach_attempts,
        "parent_changes": sample.parent_changes,
        "partitions": sample.partitions,
        "coap_sent": sample.coap_sent,
        "coap_delivered": sample.coap_delivered,
        "coap_delivery_ratio": round(sample.coap_delivery_ratio, 6),
        "coap_latency_ms_mean": sample.coap_latency_ms_mean,
        "coap_latency_ms_p95": sample.coap_latency_ms_p95,
        "deep_mle_advertisement": sample.deep_mle_advertisement,
        "deep_mle_link_request": sample.deep_mle_link_request,
        "deep_mle_link_accept": sample.deep_mle_link_accept,
        "deep_address_solicit": sample.deep_address_solicit,
        "deep_address_notify": sample.deep_address_notify,
        "deep_parent_request": sample.deep_parent_request,
        "deep_parent_response": sample.deep_parent_response,
        "baseline_mle_adv": round(baseline_mle, 6),
        "mle_adv_ratio_vs_baseline": round(mle_ratio, 6),
        "inferred_trickle_reset": int(trickle_reset),
    }


def run_timed_phase(
    ns: OTNS, bundle: NodeBundle, phase_name: str, cycle: int,
    duration_s: int, wifi_txintf: int, send_coap: bool,
    args: argparse.Namespace, coap_sender_ids: Sequence[int],
    deep_collector: DeepLogCollector, coap_fp,
    event_timeline: List[Dict[str, object]], start_time_s: int,
    prev_states: Dict[int, str],
    prev_mac: Dict[int, Dict[str, int]],
    prev_mle: Dict[int, Dict[str, int]],
    prev_radio: Dict[int, Dict[str, int]],
    baseline_mle: float, baseline_cca: float, baseline_retry: float,
) -> Tuple[int, List[Dict[str, object]], List[CounterSample],
           Dict[int, str], Dict[int, Dict[str, int]],
           Dict[int, Dict[str, int]], Dict[int, Dict[str, int]]]:

    set_wifi_interference(ns, bundle.wifi_nodes, wifi_txintf)
    event_timeline.append({
        "time_s": start_time_s, "type": "phase_start",
        "phase": phase_name, "cycle": cycle, "wifi_txintf": wifi_txintf,
    })

    rows: List[Dict[str, object]] = []
    samples: List[CounterSample] = []
    coap_window = CoapWindowStats()
    current_time = start_time_s

    for sec in range(1, duration_s + 1):
        current_time = start_time_s + sec
        if send_coap:
            senders = select_coap_senders(coap_sender_ids, args, sec)
            coap_window.sent_commands += send_periodic_coap(ns, senders, bundle.br_id, args.coap_payload_size)
        go_lines = ns.go(1)
        node_lines = deep_collector.drain_node_logs(bundle.all_thread_node_ids)
        deep_collector.feed(node_lines, event_timeline, "run", current_time)
        collect_coap_records(ns, "run", current_time, coap_fp, coap_window, coap_sender_ids)

        if sec % args.sample_interval != 0 and sec != duration_s:
            continue

        cur_states, cur_mac, cur_mle, cur_radio = collect_counter_snapshot(ns, bundle)
        deep_counts = deep_collector.drain_interval()
        sample = make_interval_sample(
            ns, bundle, cur_states, prev_states,
            cur_mac, prev_mac, cur_mle, prev_mle,
            cur_radio, prev_radio, current_time, coap_window, deep_counts,
        )
        samples.append(sample)
        trickle_reset = infer_trickle_reset(sample, baseline_mle, baseline_cca, baseline_retry, args)
        rows.append(make_row(phase_name, cycle, wifi_txintf, sample, baseline_mle, trickle_reset))
        prev_states, prev_mac, prev_mle, prev_radio = cur_states, cur_mac, cur_mle, cur_radio
        coap_window.reset()

    event_timeline.append({"time_s": current_time, "type": "phase_end", "phase": phase_name, "cycle": cycle})
    return current_time, rows, samples, prev_states, prev_mac, prev_mle, prev_radio


def run_stabilization(
    ns: OTNS, bundle: NodeBundle, args: argparse.Namespace,
    deep_collector: DeepLogCollector, coap_fp,
    event_timeline: List[Dict[str, object]], start_time_s: int,
    prev_states: Dict[int, str],
    prev_mac: Dict[int, Dict[str, int]],
    prev_mle: Dict[int, Dict[str, int]],
    prev_radio: Dict[int, Dict[str, int]],
) -> Tuple[int, bool, List[Dict[str, object]],
           Dict[int, str], Dict[int, Dict[str, int]],
           Dict[int, Dict[str, int]], Dict[int, Dict[str, int]]]:

    rows: List[Dict[str, object]] = []
    stable_streak = 0
    elapsed = 0
    coap_window = CoapWindowStats()

    while elapsed < args.stabilization_max_s:
        elapsed += 1
        t = start_time_s + elapsed
        go_lines = ns.go(1)
        node_lines = deep_collector.drain_node_logs(bundle.all_thread_node_ids)
        deep_collector.feed(node_lines, event_timeline, "run", t)
        collect_coap_records(ns, "run", t, coap_fp, coap_window, [])

        if elapsed % args.sample_interval != 0 and elapsed != args.stabilization_max_s:
            continue

        cur_states, cur_mac, cur_mle, cur_radio = collect_counter_snapshot(ns, bundle)
        deep_counts = deep_collector.drain_interval()
        sample = make_interval_sample(
            ns, bundle, cur_states, prev_states,
            cur_mac, prev_mac, cur_mle, prev_mle,
            cur_radio, prev_radio, t, coap_window, deep_counts,
        )
        stable_now = is_stable(sample, args)
        stable_streak = stable_streak + 1 if stable_now else 0
        rows.append(make_row("stabilization", 0, 0, sample, 0.0, False))

        if stable_streak >= args.stable_samples_required:
            event_timeline.append({"time_s": t, "type": "stabilization_complete", "streak": stable_streak})
            prev_states, prev_mac, prev_mle, prev_radio = cur_states, cur_mac, cur_mle, cur_radio
            coap_window.reset()
            return t, True, rows, prev_states, prev_mac, prev_mle, prev_radio

        prev_states, prev_mac, prev_mle, prev_radio = cur_states, cur_mac, cur_mle, cur_radio
        coap_window.reset()

    event_timeline.append({"time_s": start_time_s + elapsed, "type": "stabilization_timeout"})
    return start_time_s + elapsed, False, rows, prev_states, prev_mac, prev_mle, prev_radio


def compute_baseline(samples: Sequence[CounterSample]) -> Tuple[float, float, float]:
    if not samples:
        return 0.0, 0.0, 0.0
    return (
        mean(s.deep_mle_advertisement for s in samples),
        mean(s.cca_fail_ratio for s in samples),
        mean(s.retry_ratio for s in samples),
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    otns_args = ["-seed", str(seed), "-phy-tx-stats", "-pcap", "wpan-tap", "-log", args.otns_log_level]

    deep_collector = DeepLogCollector(enabled=True)
    event_timeline: List[Dict[str, object]] = []
    all_rows: List[Dict[str, object]] = []

    print(f"=== MLE Trickle Vicious Cycle Experiment ===")
    print(f"Output: {output_dir}")
    print(f"Seed: {seed}")
    print()

    with OTNS(otns_args=otns_args) as ns:
        ns.speed = args.speed
        ns.radiomodel = args.radiomodel
        if args.web:
            ns.web("stats")
        compat_watch_default(ns, "trace")
        compat_watch_all(ns, "trace")
        try:
            ns.coaps_enable()
        except OTNSCliError:
            pass

        print("[Phase 1] Building topology: 1 BR + 30 REEDs ...")
        bundle = build_reed_grid(ns, args)
        enable_candidate_router_eligibility(ns, bundle.candidate_ids)
        event_timeline.append({
            "time_s": 0, "type": "topology_built",
            "br_id": bundle.br_id, "reed_count": len(bundle.candidates),
            "wifi_count": len(bundle.wifi_nodes),
        })

        print(f"[Phase 1] Warmup: {args.warmup_s}s ...")
        safe_go(ns, args.warmup_s)
        readiness_wait = wait_for_network_readiness(
            ns, bundle, args.ready_timeout_s, args.ready_stable_s,
            event_timeline, "run", args.warmup_s,
        )
        print(f"  Network ready after {readiness_wait}s additional wait")

        reset_observation_window(ns, bundle)
        ns.kpi_start()
        deep_collector.drain_node_logs(bundle.all_thread_node_ids)
        prev_states, prev_mac, prev_mle, prev_radio = collect_counter_snapshot(ns, bundle)
        current_time = args.warmup_s + readiness_wait
        coap_sender_ids = get_coap_sender_ids(bundle, args.coap_sender_class)
        coap_log_path = output_dir / "coap_messages.jsonl"

        with coap_log_path.open("w", encoding="utf-8") as coap_fp:
            # Phase 2: Stabilization
            print(f"[Phase 2] Stabilization (max {args.stabilization_max_s}s) ...")
            (current_time, stable_ok, stab_rows,
             prev_states, prev_mac, prev_mle, prev_radio) = run_stabilization(
                ns, bundle, args, deep_collector, coap_fp,
                event_timeline, current_time,
                prev_states, prev_mac, prev_mle, prev_radio,
            )
            all_rows.extend(stab_rows)
            print(f"  Stabilization {'completed' if stable_ok else 'TIMED OUT'} at t={current_time}s")

            # Phase 3: CoAP baseline
            print(f"[Phase 3] CoAP baseline: {args.coap_baseline_s}s ...")
            (current_time, baseline_rows, baseline_samples,
             prev_states, prev_mac, prev_mle, prev_radio) = run_timed_phase(
                ns, bundle, "coap_baseline", 0, args.coap_baseline_s,
                wifi_txintf=0, send_coap=True, args=args,
                coap_sender_ids=coap_sender_ids, deep_collector=deep_collector,
                coap_fp=coap_fp, event_timeline=event_timeline,
                start_time_s=current_time,
                prev_states=prev_states, prev_mac=prev_mac,
                prev_mle=prev_mle, prev_radio=prev_radio,
                baseline_mle=0.0, baseline_cca=0.0, baseline_retry=0.0,
            )
            all_rows.extend(baseline_rows)
            baseline_mle, baseline_cca, baseline_retry = compute_baseline(baseline_samples)
            print(f"  Baseline MLE adv/interval: {baseline_mle:.1f}, CCA fail: {baseline_cca:.4f}, retry: {baseline_retry:.4f}")

            # Phase 4: Wi-Fi burst #1
            print(f"[Phase 4] Wi-Fi burst #1 (txintf={args.wifi_burst1_txintf}) for {args.wifi_burst1_s}s ...")
            (current_time, burst1_rows, burst1_samples,
             prev_states, prev_mac, prev_mle, prev_radio) = run_timed_phase(
                ns, bundle, "wifi_burst_1", 1, args.wifi_burst1_s,
                wifi_txintf=args.wifi_burst1_txintf, send_coap=True, args=args,
                coap_sender_ids=coap_sender_ids, deep_collector=deep_collector,
                coap_fp=coap_fp, event_timeline=event_timeline,
                start_time_s=current_time,
                prev_states=prev_states, prev_mac=prev_mac,
                prev_mle=prev_mle, prev_radio=prev_radio,
                baseline_mle=baseline_mle, baseline_cca=baseline_cca,
                baseline_retry=baseline_retry,
            )
            all_rows.extend(burst1_rows)
            burst1_mle = mean(s.deep_mle_advertisement for s in burst1_samples) if burst1_samples else 0
            burst1_resets = sum(1 for r in burst1_rows if r["inferred_trickle_reset"])
            print(f"  Burst #1 avg MLE adv: {burst1_mle:.1f} (ratio: {safe_ratio(burst1_mle, baseline_mle):.2f}x), trickle resets: {burst1_resets}")

            # Phase 4b: Recovery gap
            print(f"[Phase 4b] Recovery gap: {args.recovery_gap_s}s ...")
            (current_time, gap1_rows, _,
             prev_states, prev_mac, prev_mle, prev_radio) = run_timed_phase(
                ns, bundle, "recovery_gap_1", 1, args.recovery_gap_s,
                wifi_txintf=0, send_coap=True, args=args,
                coap_sender_ids=coap_sender_ids, deep_collector=deep_collector,
                coap_fp=coap_fp, event_timeline=event_timeline,
                start_time_s=current_time,
                prev_states=prev_states, prev_mac=prev_mac,
                prev_mle=prev_mle, prev_radio=prev_radio,
                baseline_mle=baseline_mle, baseline_cca=baseline_cca,
                baseline_retry=baseline_retry,
            )
            all_rows.extend(gap1_rows)

            # Phase 5: Wi-Fi burst #2
            print(f"[Phase 5] Wi-Fi burst #2 (txintf={args.wifi_burst2_txintf}) for {args.wifi_burst2_s}s ...")
            (current_time, burst2_rows, burst2_samples,
             prev_states, prev_mac, prev_mle, prev_radio) = run_timed_phase(
                ns, bundle, "wifi_burst_2", 2, args.wifi_burst2_s,
                wifi_txintf=args.wifi_burst2_txintf, send_coap=True, args=args,
                coap_sender_ids=coap_sender_ids, deep_collector=deep_collector,
                coap_fp=coap_fp, event_timeline=event_timeline,
                start_time_s=current_time,
                prev_states=prev_states, prev_mac=prev_mac,
                prev_mle=prev_mle, prev_radio=prev_radio,
                baseline_mle=baseline_mle, baseline_cca=baseline_cca,
                baseline_retry=baseline_retry,
            )
            all_rows.extend(burst2_rows)
            burst2_mle = mean(s.deep_mle_advertisement for s in burst2_samples) if burst2_samples else 0
            burst2_resets = sum(1 for r in burst2_rows if r["inferred_trickle_reset"])
            print(f"  Burst #2 avg MLE adv: {burst2_mle:.1f} (ratio: {safe_ratio(burst2_mle, baseline_mle):.2f}x), trickle resets: {burst2_resets}")

            # Phase 6: Sustained storm
            print(f"[Phase 6] Sustained storm (txintf={args.wifi_storm_txintf}) for {args.wifi_storm_s}s ...")
            (current_time, storm_rows, storm_samples,
             prev_states, prev_mac, prev_mle, prev_radio) = run_timed_phase(
                ns, bundle, "sustained_storm", 3, args.wifi_storm_s,
                wifi_txintf=args.wifi_storm_txintf, send_coap=True, args=args,
                coap_sender_ids=coap_sender_ids, deep_collector=deep_collector,
                coap_fp=coap_fp, event_timeline=event_timeline,
                start_time_s=current_time,
                prev_states=prev_states, prev_mac=prev_mac,
                prev_mle=prev_mle, prev_radio=prev_radio,
                baseline_mle=baseline_mle, baseline_cca=baseline_cca,
                baseline_retry=baseline_retry,
            )
            all_rows.extend(storm_rows)
            storm_mle = mean(s.deep_mle_advertisement for s in storm_samples) if storm_samples else 0
            storm_resets = sum(1 for r in storm_rows if r["inferred_trickle_reset"])
            print(f"  Storm avg MLE adv: {storm_mle:.1f} (ratio: {safe_ratio(storm_mle, baseline_mle):.2f}x), trickle resets: {storm_resets}")

            # Phase 7: Recovery
            print(f"[Phase 7] Recovery: {args.recovery_s}s ...")
            (current_time, recovery_rows, recovery_samples,
             prev_states, prev_mac, prev_mle, prev_radio) = run_timed_phase(
                ns, bundle, "recovery", 0, args.recovery_s,
                wifi_txintf=0, send_coap=True, args=args,
                coap_sender_ids=coap_sender_ids, deep_collector=deep_collector,
                coap_fp=coap_fp, event_timeline=event_timeline,
                start_time_s=current_time,
                prev_states=prev_states, prev_mac=prev_mac,
                prev_mle=prev_mle, prev_radio=prev_radio,
                baseline_mle=baseline_mle, baseline_cca=baseline_cca,
                baseline_retry=baseline_retry,
            )
            all_rows.extend(recovery_rows)
            recovery_mle = mean(s.deep_mle_advertisement for s in recovery_samples) if recovery_samples else 0
            print(f"  Recovery avg MLE adv: {recovery_mle:.1f}")

        ns.kpi_stop()
        ns.kpi_save(str(output_dir / "kpi.json"))
        copy_run_artifacts(output_dir)
        final_nodes = collect_final_node_metrics(ns, bundle, "run")

    write_csv(output_dir / "interval_samples.csv", all_rows)
    write_csv(output_dir / "final_node_metrics.csv", final_nodes)
    write_json(output_dir / "event_timeline.json", event_timeline)
    write_json(output_dir / "experiment_config.json", vars(args))

    # Summary
    print()
    print("=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    phase_stats: Dict[str, List[Dict[str, object]]] = {}
    for row in all_rows:
        phase_stats.setdefault(row["phase"], []).append(row)

    print(f"{'Phase':<22} {'Samples':>7} {'Avg MLE Adv':>12} {'Peak MLE':>10} {'Avg CCA%':>10} {'Resets':>7} {'MLE Ratio':>10}")
    print("-" * 80)
    for pn in ["stabilization", "coap_baseline", "wifi_burst_1", "recovery_gap_1",
                "wifi_burst_2", "sustained_storm", "recovery"]:
        prows = phase_stats.get(pn, [])
        if not prows:
            continue
        n = len(prows)
        avg_mle = mean(r["deep_mle_advertisement"] for r in prows)
        peak_mle = max(r["deep_mle_advertisement"] for r in prows)
        avg_cca = mean(r["cca_fail_ratio"] for r in prows) * 100
        resets = sum(r["inferred_trickle_reset"] for r in prows)
        ratio = safe_ratio(avg_mle, baseline_mle) if baseline_mle > 0 else 0
        print(f"{pn:<22} {n:>7} {avg_mle:>12.1f} {peak_mle:>10} {avg_cca:>9.2f}% {resets:>7} {ratio:>9.2f}x")

    total_resets = sum(r["inferred_trickle_reset"] for r in all_rows)
    vicious_cycle_detected = (
        storm_mle > baseline_mle * 2.0
        and storm_resets > 0
        and any(s.cca_fail_ratio > baseline_cca + 0.02 for s in storm_samples)
    ) if storm_samples and baseline_mle > 0 else False

    print()
    print(f"Total inferred Trickle resets: {total_resets}")
    print(f"Vicious cycle detected: {'YES' if vicious_cycle_detected else 'NO'}")
    if vicious_cycle_detected:
        print(f"  Storm MLE / Baseline MLE = {safe_ratio(storm_mle, baseline_mle):.2f}x")
        max_storm_cca = max(s.cca_fail_ratio for s in storm_samples)
        print(f"  Peak CCA failure ratio during storm: {max_storm_cca:.4f}")
        print(f"  Baseline CCA failure ratio: {baseline_cca:.4f}")
    print()
    print(f"Results saved to: {output_dir}")
    print(f"  interval_samples.csv  - per-interval time series")
    print(f"  event_timeline.json   - phase/event log")
    print(f"  kpi.json              - OTNS KPI snapshot")
    print(f"  coap_messages.jsonl   - raw CoAP messages")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLE Trickle Timer Vicious Cycle Experiment")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--seed", type=int, default=71000)
    p.add_argument("--speed", type=float, default=1000000)
    p.add_argument("--radiomodel", default="MutualInterference")
    p.add_argument("--otns-log-level", default="debug")
    p.add_argument("--web", action="store_true")

    p.add_argument("--reed-count", type=int, default=30)
    p.add_argument("--grid-spacing", type=int, default=20)
    p.add_argument("--grid-origin-x", type=int, default=20)
    p.add_argument("--grid-origin-y", type=int, default=25)
    p.add_argument("--br-offset-y", type=int, default=20)
    p.add_argument("--thread-range", type=int, default=38)
    p.add_argument("--thread-txpower", type=int, default=0)
    p.add_argument("--router-upgrade-threshold", type=int, default=16)
    p.add_argument("--router-downgrade-threshold", type=int, default=23)
    p.add_argument("--router-selection-jitter", type=int, default=120)

    p.add_argument("--warmup-s", type=int, default=60)
    p.add_argument("--ready-timeout-s", type=int, default=180)
    p.add_argument("--ready-stable-s", type=int, default=10)
    p.add_argument("--stabilization-max-s", type=int, default=180)
    p.add_argument("--stable-samples-required", type=int, default=3)
    p.add_argument("--stable-mle-threshold", type=int, default=10)
    p.add_argument("--stable-churn-threshold", type=int, default=0)
    p.add_argument("--stable-parent-threshold", type=int, default=0)
    p.add_argument("--sample-interval", type=int, default=10)

    p.add_argument("--coap-baseline-s", type=int, default=120)
    p.add_argument("--coap-period-seconds", type=int, default=1)
    p.add_argument("--coap-payload-size", type=int, default=50)
    p.add_argument("--coap-sender-class", default="router_capable")
    p.add_argument("--senders-per-tick", type=int, default=10)

    p.add_argument("--wifi-node-count", type=int, default=2)
    p.add_argument("--wifi-spacing", type=int, default=12)
    p.add_argument("--wifi-burst1-txintf", type=int, default=50)
    p.add_argument("--wifi-burst1-s", type=int, default=90)
    p.add_argument("--wifi-burst2-txintf", type=int, default=75)
    p.add_argument("--wifi-burst2-s", type=int, default=90)
    p.add_argument("--wifi-storm-txintf", type=int, default=85)
    p.add_argument("--wifi-storm-s", type=int, default=180)
    p.add_argument("--recovery-gap-s", type=int, default=60)
    p.add_argument("--recovery-s", type=int, default=120)

    p.add_argument("--reset-factor", type=float, default=3.0)
    p.add_argument("--reset-min-advertisements", type=int, default=12)
    p.add_argument("--reset-cca-delta", type=float, default=0.03)
    p.add_argument("--reset-retry-delta", type=float, default=0.05)

    return p.parse_args()


if __name__ == "__main__":
    try:
        run_experiment(parse_args())
    except OTNSExitedError as ex:
        if ex.exit_code != 0:
            raise
