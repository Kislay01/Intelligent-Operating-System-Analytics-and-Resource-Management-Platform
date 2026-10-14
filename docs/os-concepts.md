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
