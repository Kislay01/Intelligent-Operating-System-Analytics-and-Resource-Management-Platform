"""CPU scheduling simulation and recommendation."""

from .algorithms import FCFS, MLFQ, SJF, SRTF, PriorityScheduler, RoundRobin, Scheduler, compare, default_schedulers
from .burst_predictor import best_alpha, exponential_average
from .models import IDLE, Job, ScheduleResult, Segment
from .proportional import CFS, EEVDF, Lottery, Stride, cpu_share, nice_weight
from .recommender import GOALS, Recommendation, recommend, suggest_quantum

__all__ = [
    "CFS", "EEVDF", "Lottery", "Stride", "cpu_share", "nice_weight",
    "FCFS", "GOALS", "IDLE", "Job", "MLFQ", "PriorityScheduler", "Recommendation", "RoundRobin", "SJF", "SRTF",
    "ScheduleResult", "Scheduler", "Segment", "best_alpha", "compare", "default_schedulers",
    "exponential_average", "recommend", "suggest_quantum",
]
