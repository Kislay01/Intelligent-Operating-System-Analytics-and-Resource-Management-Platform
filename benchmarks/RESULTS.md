# Benchmark results

Generated 2026-10-14 08:10 on Linux 6.18.44-fc-v114 (x86_64), Python 3.13.16. Reproduce with `python benchmarks/run_all.py`; raw numbers are in `benchmarks/results/*.json`.

## Closed-loop resource allocation

A latency probe (sleeps 4 ms, then does 1 ms of work) competes with 4 CPU-bound batch processes on 2 CPUs. Wake-up latency = how late the probe runs after its timer fires. 3 trials x 8.0 s per policy; cgroup v1.

| policy | p50_ms | p95_ms | p99_ms | p99_std | p99_reduction_% | batch_throughput_% |
|---|---|---|---|---|---|---|
| default (no control) | 0.083 | 4.828 | 5.154 | 0.249 | 0.0 | 100.0 |
| static nice 19 | 0.077 | 0.099 | 0.153 | 0.062 | 97.0 | 97.7 |
| rule-based | 0.077 | 0.105 | 1.056 | 1.279 | 79.5 | 98.4 |
| Q-learning | 0.077 | 0.103 | 0.348 | 0.289 | 93.2 | 97.7 |
| Q-learning + cgroup | 0.081 | 0.111 | 2.864 | 0.072 | 44.4 | 97.7 |

`batch_throughput_%` = CPU time the batch processes received relative to no control.

## Policy learning in the calibrated simulator

Simulator calibrated against measured wake-up latency (p95 ms by batch nice: {'0': 4.83, '5': 3.743, '10': 0.11, '15': 0.104, '19': 0.106}) → preempt ratio 4.

3000 training episodes; evaluation on 300 held-out scenarios (reward = batch throughput vs. fair share minus SLO violation penalty; regret vs. the best fixed action per scenario).

| policy | reward | latency_p95 | batch_share | regret |
|---|---|---|---|---|
| Q-learning | -0.0621 | 1.7918 | 0.3601 | 0.2073 |
| LinUCB bandit | -0.252 | 2.0818 | 0.3905 | 0.3971 |
| default (nice 0) | -0.6114 | 3.16 | 0.4953 | 0.7566 |
| always nice 19 | -0.7594 | 2.3652 | 0.1883 | 0.9046 |
| rule-based | -0.7891 | 2.6827 | 0.2557 | 0.9343 |


## Collector overhead

| processes | psutil ms/sample | /proc ms/sample | speedup | recorder CPU % @2s |
|---|---|---|---|---|
| 65 | 17.69 | 7.08 | 2.5 | 0.4 |


## CPU scheduling

Averages over 5 synthetic profiles x 5 seeds (10 processes each).

| algorithm | avg_waiting | avg_response | p95_response | fairness | context_switches |
|---|---|---|---|---|---|
| SRTF | 30.02 | 26.95 | 67.36 | 0.79 | 10.28 |
| SJF | 37.33 | 37.33 | 75.41 | 0.81 | 9.0 |
| FCFS | 44.95 | 44.95 | 80.52 | 0.65 | 9.0 |
| Priority (NP) | 45.19 | 45.19 | 84.02 | 0.63 | 9.0 |
| Priority (P, aging) | 45.37 | 32.23 | 68.25 | 0.6 | 12.72 |
| EEVDF (slice=3) | 55.53 | 10.26 | 26.43 | 0.79 | 34.88 |
| Lottery (q=1) | 55.75 | 5.13 | 18.09 | 0.8 | 79.32 |
| Stride (q=1) | 56.99 | 1.86 | 4.34 | 0.86 | 92.92 |
| CFS (lat=6) | 57.02 | 2.48 | 5.37 | 0.86 | 71.36 |
| Round Robin (q=4) | 58.62 | 11.47 | 22.11 | 0.8 | 27.56 |
| MLFQ (4/8/FCFS) | 59.18 | 8.8 | 16.64 | 0.83 | 24.6 |


### Next-burst prediction (held-out sequences)

| method | MAE | improvement_vs_exp_avg_% | wins |
|---|---|---|---|
| gradient_boosting | 3.439 | 46.2 | 12 |
| exp_avg_tuned | 5.256 | 17.8 | 15 |
| exp_avg_a=0.5 | 6.396 | 0.0 | 2 |
| last_burst | 8.298 | -29.7 | 3 |


### Live burst trace capture

Captured 70 CPU bursts from 4 processes via /proc jiffy counters; a probe with known 60 ms bursts was measured at a median of 65.1 ms. Replayed as 60 jobs; recommended policy: Stride (q=1).

## Memory forecasting

40 synthetic memory series (leak, GC sawtooth, step load, daily cycle, stable); forecast horizon 30 samples. 'Auto-selected' chooses a model per series from the training part only.

| model | MAE | MAPE | mape_reduction_vs_naive_% |
|---|---|---|---|
| Holt (damped trend) | 1.115 | 2.381 | 25.1 |
| Quantile GBM | 1.2 | 2.6 | 18.2 |
| Auto-selected | 1.401 | 2.989 | 5.9 |
| Naive (last value) | 1.487 | 3.177 | 0.0 |
| Moving average (10) | 1.624 | 3.448 | -8.5 |
| Linear trend | 2.027 | 4.357 | -37.1 |


MAPE by series type:

| kind | Auto-selected | Holt (damped trend) | Linear trend | Moving average (10) | Naive (last value) | Quantile GBM |
|---|---|---|---|---|---|---|
| daily_cycle | 6.139 | 3.965 | 14.965 | 8.53 | 6.312 | 4.738 |
| gc_sawtooth | 5.505 | 4.531 | 3.689 | 4.498 | 4.609 | 4.633 |
| leak | 0.829 | 0.836 | 0.807 | 1.949 | 1.713 | 0.84 |
| stable | 1.341 | 1.244 | 1.326 | 1.322 | 1.729 | 1.637 |
| step_load | 1.131 | 1.33 | 0.999 | 0.941 | 1.524 | 1.154 |

On leaking processes the best model cuts MAPE by 52.9% versus the naive forecast.

**Leak detection** on 200 simulated processes (linear and sawtooth leaks vs stable, spiky and step-change processes): precision 0.92, recall 1.0, F1 0.958.

**OOM victim prediction**: recomputed `oom_score` matches the kernel's value (±1) for 100% of 9 user processes.

## Deadlock prediction (lock-order validation)

200 generated multi-threaded programs, half with a seeded lock-order inversion.

| lockdep detection rate | false positive rate | deadlock actually observed (short sections) | deadlock observed (2 ms sections) |
|---|---|---|---|
| 1.0 | 0.0 | 0.0 | 0.68 |


## Page replacement

Mattson single-pass LRU curve matches per-size simulation: **True** (34.1x faster for 32 frame sizes).

Hot set + sequential scans, 8 frames (faults): FIFO 1397, LRU 1397, Clock 1397, ARC 1205, Optimal 1205

Algorithmic traces, 8 frames:

| trace | refs | pages | FIFO | LRU | ARC | Optimal |
|---|---|---|---|---|---|---|
| matrix_row_major | 32 | 32 | 32 | 32 | 32 | 32 |
| matrix_col_major | 4096 | 32 | 4096 | 4096 | 4096 | 3176 |
| matmul_naive | 20000 | 20 | 2216 | 1970 | 1920 | 1316 |
| matmul_blocked | 20000 | 15 | 24 | 24 | 24 | 17 |
| binary_search | 3812 | 438 | 3600 | 3593 | 2984 | 2344 |
| quicksort | 20000 | 29 | 29 | 34 | 34 | 29 |
| linked_list | 5893 | 64 | 5288 | 5275 | 5266 | 3701 |


## Log analysis on Loghub

Drain grouping accuracy:

| dataset | lines | templates_found | templates_truth | grouping_accuracy |
|---|---|---|---|---|
| BGL | 2000 | 107 | 120 | 0.971 |
| HDFS | 2000 | 17 | 14 | 0.9975 |
| Linux | 2000 | 112 | 118 | 0.69 |


BGL anomaly detection (100 windows of 20 lines, 60 test windows):

| method | precision | recall | f1 | tp | fp | fn |
|---|---|---|---|---|---|---|
| n-gram (DeepLog rule) | 0.36 | 1.0 | 0.529 | 18 | 32 | 0 |
| Isolation Forest | 0.75 | 0.833 | 0.789 | 15 | 5 | 3 |

