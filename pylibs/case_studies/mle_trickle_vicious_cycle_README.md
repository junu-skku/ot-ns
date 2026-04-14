# MLE Trickle Timer Vicious Cycle Experiment

## Overview

This experiment demonstrates the **Wi-Fi interference -> Trickle timer reset -> MLE Advertisement explosion -> channel congestion -> more resets** vicious cycle in a Thread network.

### Hypothesis

In a dense Thread network (30 REED-capable nodes + 1 BR), Wi-Fi interference causes:
1. Packet loss/corruption on the 2.4 GHz channel
2. MLE Advertisement Trickle timers reset (RFC 6206: "inconsistency" detection)
3. All 30+ nodes simultaneously flood MLE Advertisements
4. The flood itself causes CCA failures and MAC retransmissions
5. These failures trigger further Trickle resets -> feedback loop

## Experiment Phases

| Phase | Duration | Wi-Fi txintf | Purpose |
|-------|----------|-------------|---------|
| 1. Topology build | ~60s warmup | 0 | Place 1 BR + 30 REEDs in 6x5 grid |
| 2. Stabilization | up to 180s | 0 | Wait for Trickle timers to reach maximum |
| 3. CoAP baseline | 120s | 0 | Confirm MLE stays low with app traffic only |
| 4. Wi-Fi burst #1 | 90s | 50 | Moderate interference -> initial spike |
| 4b. Recovery gap | 60s | 0 | Brief recovery between bursts |
| 5. Wi-Fi burst #2 | 90s | 75 | Stronger interference -> amplified spike |
| 6. Sustained storm | 180s | 85 | Continuous high interference -> vicious cycle |
| 7. Recovery | 120s | 0 | Interference off -> observe recovery |

## Prerequisites

- Python 3.8+
- OTNS binary in PATH (`go install ./cmd/otns`)
- `ot-rfsim` binaries (see `GUIDE.md`)

## Running

### PowerShell (Windows)

```powershell
cd pylibs/case_studies
.\run_mle_vicious_cycle.ps1
```

### Bash (Linux/macOS)

```bash
cd pylibs/case_studies
chmod +x run_mle_vicious_cycle.sh
./run_mle_vicious_cycle.sh
```

### Direct Python

```bash
cd pylibs/case_studies
python3 mle_trickle_vicious_cycle.py \
    --speed 1000000 \
    --stabilization-max-s 180 \
    --coap-baseline-s 120 \
    --wifi-burst1-s 90 --wifi-burst1-txintf 50 \
    --wifi-burst2-s 90 --wifi-burst2-txintf 75 \
    --wifi-storm-s 180 --wifi-storm-txintf 85 \
    --recovery-s 120
```

### With web visualization

```bash
python3 mle_trickle_vicious_cycle.py --web
```

## Output Files

All outputs go to `tmp/mle_trickle_vicious_cycle/`:

| File | Description |
|------|-------------|
| `interval_samples.csv` | Per-10s interval time series with all metrics |
| `event_timeline.json` | Phase transitions and detected events |
| `kpi.json` | OTNS KPI snapshot (MAC/MLE counters, CoAP stats) |
| `coap_messages.jsonl` | Raw CoAP message records |
| `final_node_metrics.csv` | Per-node final counter state |
| `experiment_config.json` | Full experiment configuration |

## Key Metrics in `interval_samples.csv`

| Column | Description |
|--------|-------------|
| `deep_mle_advertisement` | MLE Advertisement count (from trace logs) |
| `mle_adv_ratio_vs_baseline` | Ratio vs CoAP-only baseline |
| `inferred_trickle_reset` | 1 if Trickle reset was inferred |
| `cca_fail_ratio` | CCA failure ratio (channel busy) |
| `retry_ratio` | MAC retransmission ratio |
| `tx_err_cca` | Absolute CCA error count |
| `role_churn` | State changes + parent changes + attach attempts |
| `coap_delivery_ratio` | Application-layer delivery success |
| `coap_latency_ms_mean` | Mean CoAP round-trip latency |

## Interpreting Results

### Vicious cycle detected if:

1. **Storm MLE / Baseline MLE > 2x**: MLE Advertisements during sustained storm are at least double the baseline
2. **Trickle resets > 0**: At least one interval shows inferred Trickle reset
3. **CCA failure rise**: Storm CCA failure ratio exceeds baseline by > 0.02

### Expected output pattern:

```
Phase                  Samples  Avg MLE Adv   Peak MLE   Avg CCA%  Resets  MLE Ratio
--------------------------------------------------------------------------------
stabilization               X          Y.Y          Z      0.00%       0      0.00x
coap_baseline               9          3.2          5      0.12%       0      1.00x
wifi_burst_1                6         18.5         32      2.45%       3      5.78x
recovery_gap_1              4          6.1          9      0.31%       0      1.91x
wifi_burst_2                6         35.7         58      4.82%       5     11.16x
sustained_storm            12         42.3         71      6.15%      10     13.22x
recovery                    9          8.4         15      0.52%       1      2.63x
```

## Trickle Reset Inference

Since OTNS/OpenThread CLI does not expose the Trickle timer value directly, we infer resets conservatively:

- **MLE burst**: `deep_mle_advertisement >= max(12, baseline * 3.0)`
- **AND** at least one of:
  - CCA failure ratio rise >= 0.03 above baseline
  - Retry ratio rise >= 0.05 above baseline

## Tuning Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--wifi-burst1-txintf` | 50 | Moderate interference strength |
| `--wifi-burst2-txintf` | 75 | Strong interference strength |
| `--wifi-storm-txintf` | 85 | Maximum interference for vicious cycle |
| `--reset-factor` | 3.0 | MLE burst must be Nx baseline to infer reset |
| `--reset-min-advertisements` | 12 | Minimum absolute MLE count for reset inference |
| `--grid-spacing` | 20 | Node spacing (smaller = denser = more contention) |
| `--wifi-node-count` | 2 | Number of Wi-Fi interferers |
