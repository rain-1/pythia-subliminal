from __future__ import annotations

import random
from collections import deque
from typing import Iterable

import numpy as np


def additive_design_matrix(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        raise ValueError(f"grid must be square, got shape={grid.shape}")
    n = grid.shape[0]
    rows = []
    for i, j in np.argwhere(grid == 1):
        row = np.zeros(2 * n - 1, dtype=float)
        row[0] = 1.0
        if i > 0:
            row[i] = 1.0
        if j > 0:
            row[n + j - 1] = 1.0
        rows.append(row)
    return np.vstack(rows) if rows else np.empty((0, 2 * n - 1), dtype=float)


def check_full_rank(grid: np.ndarray) -> None:
    n = int(np.asarray(grid).shape[0])
    x = additive_design_matrix(grid)
    rank = int(np.linalg.matrix_rank(x))
    expected = 2 * n - 1
    if rank != expected:
        raise ValueError(f"Design matrix is rank deficient: rank={rank}, expected={expected}")


def _check_connected(grid: np.ndarray) -> None:
    n = int(grid.shape[0])
    adjacency = {("r", i): [] for i in range(n)}
    adjacency.update({("c", j): [] for j in range(n)})
    for i, j in np.argwhere(grid == 1):
        r = ("r", int(i))
        c = ("c", int(j))
        adjacency[r].append(c)
        adjacency[c].append(r)

    start = ("r", 0)
    seen = {start}
    queue: deque[tuple[str, int]] = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in adjacency[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    expected = 2 * n
    if len(seen) != expected:
        raise ValueError(f"Bipartite graph is disconnected: reached={len(seen)}, expected={expected}")


def validate_balanced_grid(
    grid: np.ndarray,
    n: int,
    k: int,
    require_connected: bool = True,
    require_full_rank: bool = True,
) -> None:
    grid = np.asarray(grid)
    if grid.shape != (n, n):
        raise ValueError(f"grid has shape {grid.shape}, expected {(n, n)}")
    if not np.isin(grid, [0, 1]).all():
        raise ValueError("grid must be binary")
    if not np.all(np.diag(grid) == 1):
        raise ValueError("full diagonal must be included")

    row_sums = grid.sum(axis=1)
    col_sums = grid.sum(axis=0)
    if not np.all(row_sums == k + 1):
        raise ValueError(f"row sums must all equal {k + 1}: {row_sums.tolist()}")
    if not np.all(col_sums == k + 1):
        raise ValueError(f"column sums must all equal {k + 1}: {col_sums.tolist()}")

    off = grid.copy()
    np.fill_diagonal(off, 0)
    off_row_sums = off.sum(axis=1)
    off_col_sums = off.sum(axis=0)
    if not np.all(off_row_sums == k):
        raise ValueError(f"off-diagonal row sums must all equal {k}: {off_row_sums.tolist()}")
    if not np.all(off_col_sums == k):
        raise ValueError(f"off-diagonal column sums must all equal {k}: {off_col_sums.tolist()}")

    if require_connected:
        _check_connected(grid)
    if require_full_rank:
        check_full_rank(grid)


def _cyclic_grid(n: int, k: int) -> tuple[np.ndarray, set[tuple[int, int]], list[tuple[int, int]]]:
    if n <= 1:
        raise ValueError("n must be greater than 1")
    if k < 0 or k >= n:
        raise ValueError(f"k must satisfy 0 <= k < n, got n={n}, k={k}")
    grid = np.zeros((n, n), dtype=np.int8)
    off_set: set[tuple[int, int]] = set()
    off_list: list[tuple[int, int]] = []
    for i in range(n):
        grid[i, i] = 1
        for d in range(1, k + 1):
            cell = (i, (i + d) % n)
            grid[cell] = 1
            off_set.add(cell)
            off_list.append(cell)
    return grid, off_set, off_list


def _replace_cell(cells: list[tuple[int, int]], old: tuple[int, int], new: tuple[int, int]) -> None:
    idx = cells.index(old)
    cells[idx] = new


def _randomize_by_edge_swaps(
    grid: np.ndarray,
    off_set: set[tuple[int, int]],
    off_list: list[tuple[int, int]],
    swaps: int,
    rng: random.Random,
) -> None:
    if len(off_list) < 2:
        return
    for _ in range(swaps):
        (r1, c1), (r2, c2) = rng.sample(off_list, 2)
        if r1 == r2 or c1 == c2:
            continue
        a = (r1, c2)
        b = (r2, c1)
        if a == b or a[0] == a[1] or b[0] == b[1]:
            continue
        if a in off_set or b in off_set:
            continue

        grid[r1, c1] = 0
        grid[r2, c2] = 0
        grid[a] = 1
        grid[b] = 1
        off_set.remove((r1, c1))
        off_set.remove((r2, c2))
        off_set.add(a)
        off_set.add(b)
        _replace_cell(off_list, (r1, c1), a)
        _replace_cell(off_list, (r2, c2), b)


def make_balanced_random_grid(
    n: int,
    k: int,
    swaps: int = 5000,
    seed: int | None = None,
    require_connected: bool = True,
    require_full_rank: bool = True,
    max_attempts: int = 100,
) -> np.ndarray:
    master = random.Random(seed)
    errors: list[str] = []
    for attempt in range(max_attempts):
        attempt_seed = master.randrange(2**63) if seed is not None else None
        rng = random.Random(attempt_seed)
        grid, off_set, off_list = _cyclic_grid(n, k)
        _randomize_by_edge_swaps(grid, off_set, off_list, swaps, rng)
        try:
            validate_balanced_grid(grid, n, k, require_connected=require_connected, require_full_rank=require_full_rank)
            return grid
        except ValueError as exc:
            errors.append(f"attempt {attempt + 1}: {exc}")
    raise ValueError("Failed to generate valid balanced grid: " + "; ".join(errors[-5:]))


def make_many_balanced_random_grids(
    n: int,
    k: int,
    num_grids: int,
    swaps: int = 5000,
    seed: int | None = None,
) -> list[np.ndarray]:
    rng = random.Random(seed)
    grids = []
    for _ in range(num_grids):
        derived_seed = rng.randrange(2**63)
        grid = make_balanced_random_grid(n, k, swaps=swaps, seed=derived_seed)
        validate_balanced_grid(grid, n, k)
        grids.append(grid)
    return grids


def plot_grids(grids: Iterable[np.ndarray], title: str | None = None):
    import matplotlib.pyplot as plt

    grids = list(grids)
    fig, axes = plt.subplots(1, len(grids), figsize=(3.2 * len(grids), 3.2), squeeze=False)
    for ax, grid in zip(axes[0], grids):
        ax.imshow(grid, cmap="Greys", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"rank {np.linalg.matrix_rank(additive_design_matrix(grid))}")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    demo_grids = make_many_balanced_random_grids(20, 5, 5, swaps=5000, seed=123)
    for idx, demo_grid in enumerate(demo_grids):
        validate_balanced_grid(demo_grid, 20, 5)
        print(f"grid {idx}: row sums={demo_grid.sum(axis=1).tolist()}")
        print(f"grid {idx}: col sums={demo_grid.sum(axis=0).tolist()}")
        print(f"grid {idx}: rank={np.linalg.matrix_rank(additive_design_matrix(demo_grid))}")
    plot_grids(demo_grids, title="Balanced random grids")
    plt.show()
