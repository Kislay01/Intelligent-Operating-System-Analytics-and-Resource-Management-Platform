# OS concepts behind each module

Short notes connecting every module to the operating-system theory it implements, with the formulas and the
questions they answer. Useful for the report and the viva.

---

## 1. Process monitoring

**Process Control Block (PCB).** The kernel keeps per-process state (PID, parent, state, registers, scheduling
information, memory maps, open files). Linux exposes it through `/proc/[pid]/`.

**Process states** (`/proc/[pid]/stat`, field 3):

| Code | State | Meaning |
|---|---|---|
| R | Running / runnable | on a CPU or in the run queue |
| S | Interruptible sleep | waiting for an event; signals wake it |
| D | Uninterruptible sleep | waiting inside the kernel (usually I/O); cannot be killed |
| Z | Zombie | exited, but the parent has not called `wait()` |
| T | Stopped | suspended by a signal (SIGSTOP) or a debugger |
| I | Idle | idle kernel thread |

**Process tree.** Every process except PID 1 is created by `fork()` from a parent and usually replaced by `exec()`.
When a parent dies first, the child is an **orphan** and is re-parented to init (PID 1).

**Context switches.** `/proc/[pid]/status` reports
`voluntary_ctxt_switches` (the process blocked and gave up the CPU) and `nonvoluntary_ctxt_switches` (the scheduler
preempted it when its time slice ended). This ratio is the basis of the CPU-bound / I/O-bound classification.

**Page faults.** *Minor* faults are satisfied without disk I/O (page already in memory, e.g. page cache); *major*
faults require reading the page from disk.

**Load average.** Exponentially damped average of the number of tasks that are runnable (R) or in uninterruptible
sleep (D) over 1, 5 and 15 minutes.

**Pressure Stall Information (PSI).** `/proc/pressure/{cpu,memory,io}` reports the share of time in which *some*
(at least one) or *full* (all non-idle) tasks were stalled waiting for that resource. Unlike utilisation, it measures
contention directly.

---

## 2. CPU scheduling

**Metrics.**

* Turnaround time = completion − arrival
* Waiting time = turnaround − burst
* Response time = first time on CPU − arrival
* Throughput = completed processes / elapsed time
* CPU utilisation = busy time / elapsed time
* Jain's fairness index = (Σxᵢ)² / (n · Σxᵢ²), with xᵢ = turnaround / burst (slowdown); 1.0 = perfectly fair

**Policies.**

| Policy | Preemptive | Strength | Weakness |
|---|---|---|---|
| FCFS | no | simple, no starvation | convoy effect: short jobs wait behind long ones |
| SJF | no | optimal average waiting time (all jobs present) | needs the next burst length, starves long jobs |
| SRTF | yes | optimal average waiting time with arrivals | needs burst prediction, many switches |
| Round Robin | yes | good response time, fair | performance depends on quantum; too small → overhead |
| Priority | either | expresses importance | starvation unless **aging** raises waiting jobs' priority |
| MLFQ | yes | adapts: interactive jobs stay high, CPU hogs sink | needs tuning; periodic boost avoids starvation |

**Choosing the quantum.** Rule of thumb: about 80% of CPU bursts should be shorter than the quantum. A very large
quantum degenerates into FCFS; a very small one spends most time context switching.

**Burst prediction.** τₙ₊₁ = α·tₙ + (1 − α)·τₙ. With α = 0.5, recent and past history are weighted equally. This is
how SJF can be approximated in a real kernel.

**Linux.** Normal tasks use CFS (EEVDF since Linux 6.6), which tracks each task's virtual runtime weighted by its nice
value. Real-time tasks use SCHED_FIFO / SCHED_RR; SCHED_BATCH marks throughput-oriented work.

---

## 3. Memory management and prediction

**Virtual memory.** Each process has its own address space mapped to physical frames by page tables. Pages can be
in RAM, in swap or not yet allocated.

**RSS vs VSZ.** VSZ is all virtual memory a process has mapped; RSS is the part resident in physical RAM.

**Page cache.** File data cached in otherwise unused RAM; reclaimable, so "available" memory is larger than "free".

**OOM killer.** When memory and swap are exhausted and nothing can be reclaimed, the kernel kills the process with the
highest `oom_score` (mostly proportional to its memory usage).

**Forecasting.** Models are compared on a held-out window:

* naive (last value) — the baseline every model must beat
* moving average — smooths noise, no trend
* linear trend — least squares on recent points; captures steady growth (leaks)
* Holt's damped trend — exponential smoothing of level and trend

Error metrics: MAE = mean |error|, RMSE = √mean(error²), MAPE = mean |error / actual| × 100.

**Leak detection.** A leak is a resident set that keeps growing: positive slope, high R² (the growth is steady) and
mostly non-decreasing samples.

---

## 4. Deadlocks

**Coffman conditions** (all four are required): mutual exclusion, hold and wait, no preemption, circular wait.

**Strategies.** Prevention (break a condition, e.g. lock ordering), avoidance (Banker's algorithm), detection and
recovery (find cycles, kill or roll back), or ignore (what most general-purpose OSes do for user processes).

**Banker's algorithm.**

* Need = Max − Allocation
* Safety: Work = Available; repeatedly find a process with Need ≤ Work, let it finish, Work += Allocation.
  If every process finishes, the state is **safe** and the order is a safe sequence.
* Request: if Request ≤ Need and Request ≤ Available, pretend to grant it and run the safety check; grant only if
  the new state is safe.

An unsafe state is not necessarily deadlocked, but the system can no longer guarantee to avoid deadlock.

**Resource allocation graph.** Request edges P → R, assignment edges R → P. With single-instance resources a cycle
means deadlock. With multiple instances a cycle is only necessary; the detection algorithm (graph reduction) decides.

**Wait-for graph.** Collapse resources: Pᵢ → Pⱼ if Pᵢ waits for a resource held by Pⱼ.

**Risk score.** Prediction means flagging states that move toward deadlock before a cycle closes: unsafe state,
scarce resources, hold-and-wait and long waiting chains all raise the score.

**On Linux.** `/proc/locks` lists file locks; lines marked `->` are processes blocked on the lock above. POSIX
`fcntl` locks detect deadlocks (`EDEADLK`); `flock` locks do not, which is what the demo workload exploits.

---

## 5. Page replacement

**Demand paging.** Pages are loaded only when referenced. A reference to a non-resident page is a page fault; if no
frame is free, a victim page is evicted.

| Policy | Victim | Notes |
|---|---|---|
| FIFO | oldest loaded page | simple; suffers Belady's anomaly |
| Optimal (MIN) | page used farthest in the future | lower bound; needs future knowledge |
| LRU | least recently used page | good approximation of Optimal; expensive in hardware |
| LFU | least frequently used page | keeps hot pages; slow to adapt to phase changes |
| Clock | first page with reference bit 0 (bits cleared as the hand passes) | LRU approximation using the hardware accessed bit |

**Belady's anomaly.** With FIFO, adding frames can increase faults (e.g. `1 2 3 4 1 2 5 1 2 3 4 5`: 9 faults with 3
frames, 10 with 4). LRU and Optimal are *stack algorithms* — the pages kept with n frames are always a subset of
those kept with n + 1 — so they never show it.

**Working set and thrashing.** The working set W(t, Δ) is the set of pages referenced in the last Δ references. If the
sum of working sets exceeds available frames, processes fault constantly and spend more time paging than computing.

---

## 6. Logging

**Kernel ring buffer.** The kernel logs into a fixed-size buffer read with `dmesg`; timestamps are seconds since
boot.

**syslog / journald.** User-space daemons log through syslog or systemd-journald with a priority from 0 (emerg) to 7
(debug).

**OS events detected.** OOM kills, kernel panics/oops, soft/hard lockups, hung tasks (D state > 120 s), segmentation
faults, block-layer I/O errors, thermal throttling, service failures, authentication failures, memory pressure.

**Anomaly detection.** Lines are grouped into windows (e.g. one minute). An Isolation Forest isolates windows whose
feature vector (volume, errors, warnings, critical lines, sources, OS events) is unusual; a robust z-score
(median/MAD) on error counts gives an explainable second signal.

---

## 7. Resource allocation

**Linux levers.**

| Lever | Effect | Permission |
|---|---|---|
| `nice` (−20…19) | CFS weight; each step ≈ 10% change in CPU share | raising: own processes; lowering: root |
| CPU affinity | restricts which cores a task may run on (`sched_setaffinity`) | own processes |
| `ionice` | I/O scheduling class / priority (best-effort 0–7, idle) | own processes |
| scheduling class | SCHED_OTHER / BATCH / IDLE / FIFO / RR | real-time classes need root |
| cgroups v2 | hard and soft limits (`cpu.max`, `memory.high`, …) | delegated cgroup or root |

**Policy.** CPU-bound background work competing with interactive tasks gets a higher nice value and is pinned to a
subset of cores; I/O-bound work gets a lower I/O priority; memory-heavy processes are flagged as OOM candidates;
interactive tasks are left at full priority.

**Classifier.** A Random Forest over behavioural features. Training data is generated from the OS behaviour that
defines each class; evaluation uses stratified 5-fold cross-validation, and feature importances explain the model.

---

# Advanced topics

## A. Reading the kernel directly

| Quantity | Source | Computation |
|---|---|---|
| Process CPU% | `/proc/[pid]/stat` fields 14-15 (`utime`, `stime`, jiffies) | Δ(utime+stime) / CLK_TCK / Δwall × 100 |
| System CPU% | `/proc/stat` `cpu` line | 1 − Δ(idle+iowait) / Δtotal |
| Memory used | `/proc/meminfo` | MemTotal − MemAvailable |
| RSS / PSS / USS | `/proc/[pid]/smaps_rollup` | PSS splits shared pages; USS = private clean + private dirty |
| Kernel VM events | `/proc/vmstat` | `pgmajfault`, `pswpin/out`, `oom_kill` counters |
| Contention | `/proc/pressure/{cpu,memory,io}` | share of time tasks were stalled (some/full) |

Accounting granularity is one jiffy (1/CLK_TCK s, usually 10 ms), which bounds the resolution of burst tracing.

## B. CFS and EEVDF

**CFS** (Linux 2.6.23-6.5) keeps runnable tasks in a red-black tree ordered by *virtual runtime*. Running for Δt
adds Δt × 1024 / weight to a task's vruntime, where weight comes from the nice value (nice 0 = 1024; each nice step
changes CPU share by about 10%, the table ratio being 1.25). The scheduler always runs the leftmost (smallest
vruntime) task. The slice is sched_latency × weight / Σweights, bounded below by min_granularity. Waking tasks are
placed near min_vruntime with a capped sleeper credit, so interactive tasks run soon after waking.

**EEVDF** (Linux ≥ 6.6) replaced CFS's heuristics with a cleaner rule. A task is *eligible* if its lag ≥ 0
(it has received no more than its fair share). Among eligible tasks, it picks the one with the earliest *virtual
deadline* = vruntime + slice / weight.

**Stride and Lottery scheduling** are the textbook proportional-share schedulers. Stride (deterministic) runs the
task with the smallest *pass* and adds stride = BIG / tickets. Lottery (randomised) draws a ticket each quantum.

## C. Lock-order validation (lockdep)

Linux's lockdep records, for every *lock class*, the set of locks held when it is acquired. It builds a directed
graph of "A held while acquiring B". A cycle in that graph means two code paths can acquire the same locks in
opposite orders, and that is a *possible* deadlock even if the timing that triggers it has never occurred.
Detection costs one graph-reachability check per acquisition and finds bugs that testing almost never triggers.

## D. ARC, WSClock and Mattson

* **ARC** keeps two LRU lists of resident pages: T1 (seen once, recency) and T2 (seen at least twice, frequency).
  It also keeps two *ghost* lists, B1 and B2, of recently evicted page numbers. A hit in B1 grows the target size
  of T1; a hit in B2 shrinks it. ARC therefore tunes itself and resists scans, which flush plain LRU.
* **WSClock** runs the clock hand over the frames but evicts only pages whose reference bit is clear *and* whose
  last use is older than the working-set window τ.
* **Mattson's stack algorithm.** For a stack algorithm such as LRU, the memory contents with n frames are a subset
  of those with n + 1 frames. A reference with stack distance d therefore hits exactly when there are at least d
  frames. One pass that computes stack distances yields the fault count for every memory size: the miss-ratio
  curve.

## E. Log template mining

**Drain** stores templates in a fixed-depth tree keyed by token count and the first tokens. A new line joins the
most similar template in its leaf if at least `st` of its tokens match; mismatching positions become `<*>`.
Variables such as IPs, numbers and hex are masked first.

**Grouping accuracy** is the fraction of lines whose predicted group contains exactly the lines of their
ground-truth event.

**DeepLog** treats the template sequence as a language: it predicts the next template from the previous h, and
flags a line whose template is not among the top-k predictions.

## F. cgroups

Control groups apply limits and weights to sets of processes. In **cgroup v2** (a unified hierarchy):

* `cpu.weight` (1-10000, default 100) sets the share of CPU under contention.
* `cpu.max` ("quota period") is a hard cap.
* `memory.high` throttles and reclaims; `memory.max` invokes the OOM killer inside the group.

**cgroup v1** uses separate hierarchies per controller (`cpu.shares`, `cpu.cfs_quota_us`, `memory.limit_in_bytes`).

## G. Learning an allocation policy

The real system is slow and noisy to experiment on, so policies are learned in a simulator and then validated on
the machine (*sim-to-real*):

1. **Model.** A tick-based CFS: one interactive task (periodic bursts), batch hogs (controlled nice level), and
   other services.
2. **Calibration.** Measure the probe's p95 wake-up latency on the real machine at each batch nice level, then fit
   the simulator's wake-up-preemption parameter to match.
3. **Reward.** Batch throughput relative to its fair share, minus a penalty when interactive latency exceeds the SLO.
4. **Learners.** A LinUCB contextual bandit (ridge regression per action plus an upper-confidence exploration
   bonus) and tabular Q-learning (each scenario is a one-step episode, so Q(s, a) is a running average of rewards).
5. **Deployment.** A closed loop of observe → decide → act → measure, with a PSI-based rollback guard.
