"""Realistic page reference traces.

1. Algorithmic traces: the exact sequence of pages touched by common
   algorithms, computed from the addresses of the array elements they access
   (8-byte elements, 4 KiB pages). These show the effect of access order on
   locality, e.g. row-major vs column-major traversal of the same matrix.
2. Real program traces: Valgrind's ``lackey`` tool logs every memory access of
   a running binary (``valgrind --tool=lackey --trace-mem=yes``). The parser
   converts it to page numbers.
"""

from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path

PAGE = 4096
ELEM = 8


def _page(base: int, index: int) -> int:
    return (base + index * ELEM) // PAGE


def _compress(pages: list[int]) -> list[int]:
    """Drop immediate repeats (same page touched again before any other page): they never fault."""
    out: list[int] = []
    for p in pages:
        if not out or out[-1] != p:
            out.append(p)
    return out


def matrix_traversal(n: int = 256, order: str = "row") -> list[int]:
    """Sum an n x n double matrix stored row-major, traversing by row or by column."""
    refs = []
    for a in range(n):
        for b in range(n):
            i, j = (a, b) if order == "row" else (b, a)
            refs.append(_page(0, i * n + j))
    return _compress(refs)


def matmul(n: int = 64, blocked: int | None = None) -> list[int]:
    """C = A x B (row-major), naive ijk or tiled with block size ``blocked``."""
    a_base, b_base, c_base = 0, n * n * ELEM, 2 * n * n * ELEM
    refs = []
    if not blocked:
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    refs += [_page(a_base, i * n + k), _page(b_base, k * n + j)]
                refs.append(_page(c_base, i * n + j))
    else:
        bs = blocked
        for ii in range(0, n, bs):
            for jj in range(0, n, bs):
                for kk in range(0, n, bs):
                    for i in range(ii, min(ii + bs, n)):
                        for j in range(jj, min(jj + bs, n)):
                            for k in range(kk, min(kk + bs, n)):
                                refs += [_page(a_base, i * n + k), _page(b_base, k * n + j)]
                            refs.append(_page(c_base, i * n + j))
    return _compress(refs)


def binary_searches(n: int = 1 << 18, queries: int = 400, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    refs = []
    for _ in range(queries):
        target = rng.randrange(n)
        lo, hi = 0, n - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            refs.append(_page(0, mid))
            if mid == target:
                break
            lo, hi = (mid + 1, hi) if mid < target else (lo, mid - 1)
    return _compress(refs)


def quicksort(n: int = 1 << 15, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    arr = list(range(n))
    rng.shuffle(arr)
    refs: list[int] = []

    def part(lo, hi):
        pivot = arr[hi]
        refs.append(_page(0, hi))
        i = lo
        for j in range(lo, hi):
            refs.append(_page(0, j))
            if arr[j] < pivot:
                arr[i], arr[j] = arr[j], arr[i]
                refs.append(_page(0, i))
                i += 1
        arr[i], arr[hi] = arr[hi], arr[i]
        return i

    stack = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if lo < hi:
            p = part(lo, hi)
            stack += [(lo, p - 1), (p + 1, hi)]
    return _compress(refs)


def linked_list_walk(nodes: int = 4096, steps: int = 6000, seed: int = 0) -> list[int]:
    """Pointer chasing through nodes scattered across memory (poor locality)."""
    rng = random.Random(seed)
    order = list(range(nodes))
    rng.shuffle(order)
    nxt = {order[i]: order[(i + 1) % nodes] for i in range(nodes)}
    node, refs = order[0], []
    for _ in range(steps):
        refs.append(_page(0, node * 8))  # 64-byte nodes
        node = nxt[node]
    return _compress(refs)


ALGORITHMIC = {
    "matrix_row_major": ("Matrix sum, row-major traversal (sequential)", lambda: matrix_traversal(128, "row")),
    "matrix_col_major": ("Matrix sum, column-major traversal (strided)", lambda: matrix_traversal(128, "col")),
    "matmul_naive": ("Matrix multiply, naive ijk", lambda: matmul(96)),
    "matmul_blocked": ("Matrix multiply, 16x16 tiles", lambda: matmul(96, 16)),
    "binary_search": ("400 binary searches on a 2 MiB array", lambda: binary_searches()),
    "quicksort": ("Quicksort of 16K integers", lambda: quicksort(1 << 14)),
    "linked_list": ("Pointer chasing through a shuffled linked list", lambda: linked_list_walk()),
}


def algorithmic(name: str, limit: int | None = 20000) -> list[int]:
    refs = ALGORITHMIC[name][1]()
    return refs[:limit] if limit else refs


# ------------------------------------------------------------- valgrind lackey
def parse_lackey(text: str, include_instructions: bool = False, limit: int | None = None) -> list[int]:
    """Parse ``valgrind --tool=lackey --trace-mem=yes`` output into page numbers.

    Lines look like ``I  04016c30,3`` (instruction fetch), `` L 0021ec20,8``
    (load), `` S ...`` (store) and `` M ...`` (modify).
    """
    pages = []
    for line in text.splitlines():
        if len(line) < 4 or line.startswith("=="):
            continue
        kind = line[:2].strip()
        if kind not in ("I", "L", "S", "M") or (kind == "I" and not include_instructions):
            continue
        try:
            addr = int(line[3:].split(",")[0].strip(), 16)
        except ValueError:
            continue
        pages.append(addr // PAGE)
    pages = _compress(pages)
    return pages[:limit] if limit else pages


def trace_command(cmd: list[str], limit: int = 20000, timeout: float = 120) -> list[int]:
    """Run a command under Valgrind lackey and return its data page trace."""
    if not shutil.which("valgrind"):
        raise RuntimeError("valgrind is not installed (sudo apt install valgrind)")
    proc = subprocess.run(["valgrind", "--tool=lackey", "--trace-mem=yes", "--log-fd=2", *cmd],
                          capture_output=True, text=True, timeout=timeout, check=False)
    return parse_lackey(proc.stderr, limit=limit)


def save(refs: list[int], path: str | Path) -> Path:
    path = Path(path)
    path.write_text(" ".join(map(str, refs)))
    return path
