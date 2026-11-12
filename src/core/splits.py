"""Leakage-free trajectory segmentation, lineage grouping, and CV splitters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def segment_concatenated_array(
    X: np.ndarray,
    jump_threshold: float,
    norm: str = "inf",
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a concatenated 2D trajectory array strictly at step discontinuities.

    Parameters
    ----------
    X : np.ndarray of shape (N, d)
        Time-ordered feature matrix that may contain concatenation seams.
    jump_threshold : float
        Maximum allowed step norm ||X[t+1] - X[t]|| within a continuous segment.
        Any step strictly exceeding ``jump_threshold`` starts a new segment ID.
    norm : {"inf", "l2"}
        Step norm definition (default "inf": max_j |X[t+1, j] - X[t, j]|).

    Returns
    -------
    seg_id : np.ndarray of shape (N,), dtype int64
        Monotonically non-decreasing contiguous segment IDs in {0, ..., M-1}.
    jump_indices : np.ndarray of shape (M-1,), dtype int64
        Frame indices ``t`` where ``X[t]`` is the first frame of a new segment.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim != 2 or X_arr.shape[0] == 0:
        raise ValueError(f"Expected non-empty 2D array X, got shape {X_arr.shape}.")
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("Input X contains NaN or Inf values; audit and clean before segmentation.")
    if jump_threshold <= 0.0:
        raise ValueError(f"jump_threshold must be positive, got {jump_threshold}.")

    if X_arr.shape[0] == 1:
        return np.zeros(1, dtype=np.int64), np.empty(0, dtype=np.int64)

    diffs = np.abs(np.diff(X_arr, axis=0))
    if norm == "inf":
        step_size = np.max(diffs, axis=1)
    elif norm == "l2":
        step_size = np.linalg.norm(diffs, axis=1)
    else:
        raise ValueError(f"Unsupported norm '{norm}'; expected 'inf' or 'l2'.")

    is_jump = step_size > jump_threshold
    jump_indices = np.flatnonzero(is_jump).astype(np.int64) + 1
    seg_id = np.zeros(X_arr.shape[0], dtype=np.int64)
    if jump_indices.size > 0:
        seg_id[jump_indices] = 1
        seg_id = np.cumsum(seg_id, dtype=np.int64)
    return seg_id, jump_indices


def assert_contiguous_segments(seg_id: np.ndarray) -> None:
    """Verify that each unique segment ID appears as one contiguous block."""
    seg = np.asarray(seg_id)
    if seg.ndim != 1 or seg.size == 0:
        raise ValueError("seg_id must be a non-empty 1D array.")
    transitions = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    block_starts = np.concatenate(([0], transitions))
    block_ids = seg[block_starts]
    if np.unique(block_ids).size != block_ids.size:
        raise AssertionError(
            "Non-contiguous segment IDs detected: frames with identical seg_id "
            "are interleaved across non-adjacent sections of the array."
        )


def assert_no_cross_seam_pairs(
    seg_id: np.ndarray,
    idx_start: np.ndarray,
    idx_end: np.ndarray,
) -> None:
    """Assert that zero transition/history pairs cross trajectory/segment seams."""
    seg = np.asarray(seg_id)
    s0 = seg[np.asarray(idx_start, dtype=np.int64)]
    s1 = seg[np.asarray(idx_end, dtype=np.int64)]
    n_bad = int(np.sum(s0 != s1))
    if n_bad > 0:
        raise AssertionError(f"Detected {n_bad} cross-seam transition pairs!")


def build_contiguous_macro_groups(
    seg_id: np.ndarray,
    n_macro_blocks: int = 25,
    buffer_segments: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign contiguous trajectory segments into macro-blocks with boundary buffers.

    Segments at the boundary between adjacent macro-blocks can optionally be marked
    with ``valid_mask = False`` (``group_id = -1``) to suppress residual adaptive-round
    ordering correlation across fold boundaries.

    Returns
    -------
    group_id : np.ndarray of shape (N,), dtype int64
        Macro-block ID in {0, ..., n_eff - 1}, or -1 for excluded buffer frames.
    valid_mask : np.ndarray of shape (N,), dtype bool
        True for frames retained after boundary buffer exclusion.
    """
    assert_contiguous_segments(seg_id)
    seg = np.asarray(seg_id, dtype=np.int64)
    unique_segs = np.unique(seg)
    # Preserve appearance order of segments
    _, first_idx = np.unique(seg, return_index=True)
    ordered_segs = seg[np.sort(first_idx)]
    n_segs = ordered_segs.size

    n_blocks = max(2, min(int(n_macro_blocks), n_segs))
    chunks = np.array_split(ordered_segs, n_blocks)

    seg_to_group = {}
    for b_idx, chunk in enumerate(chunks):
        if chunk.size == 0:
            continue
        if buffer_segments > 0 and b_idx < len(chunks) - 1 and chunk.size > 2 * buffer_segments + 1:
            keep = chunk[:-buffer_segments]
            drop = chunk[-buffer_segments:]
        else:
            keep = chunk
            drop = np.empty(0, dtype=np.int64)
        for s in keep:
            seg_to_group[int(s)] = b_idx
        for s in drop:
            seg_to_group[int(s)] = -1

    group_id = np.array([seg_to_group[int(s)] for s in seg], dtype=np.int64)
    valid_mask = group_id >= 0
    return group_id, valid_mask


def build_lineage_connected_components(
    lineage_labels: Sequence,
    initial_rmsd_matrix: Optional[np.ndarray] = None,
    rmsd_cutoff: float = 1.0,
) -> np.ndarray:
    """Group trajectories into connected components by seeding lineage + initial RMSD.

    Two trajectories m and m' belong to the same atomic CV group if they share the
    same adaptive seeding label (e.g. ``pr_<round>``) OR have initial heavy-atom
    transmembrane RMSD strictly below ``rmsd_cutoff`` (in Å).

    Returns
    -------
    component_ids : np.ndarray of shape (M,), dtype int64
        Connected-component group IDs for the M trajectories.
    """
    labels = list(lineage_labels)
    m = len(labels)
    if m == 0:
        return np.empty(0, dtype=np.int64)

    rows = []
    cols = []
    by_label: dict = {}
    for idx, lbl in enumerate(labels):
        by_label.setdefault(lbl, []).append(idx)
    for members in by_label.values():
        first = members[0]
        for other in members[1:]:
            rows.extend([first, other])
            cols.extend([other, first])

    if initial_rmsd_matrix is not None:
        rmsd = np.asarray(initial_rmsd_matrix, dtype=np.float64)
        if rmsd.shape != (m, m):
            raise ValueError(f"Expected ({m}, {m}) initial_rmsd_matrix, got {rmsd.shape}.")
        close_i, close_j = np.where((rmsd < rmsd_cutoff) & (~np.eye(m, dtype=bool)))
        rows.extend(close_i.tolist())
        cols.extend(close_j.tolist())

    data = np.ones(len(rows), dtype=np.int8)
    adj = csr_matrix((data, (rows, cols)), shape=(m, m))
    _, comp_ids = connected_components(adj, directed=False, return_labels=True)
    return comp_ids.astype(np.int64)


def assert_disjoint_group_split(
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Assert zero shared trajectory/lineage groups and zero shared indices across folds."""
    tr = np.asarray(train_idx, dtype=np.int64)
    te = np.asarray(test_idx, dtype=np.int64)
    if np.intersect1d(tr, te).size > 0:
        raise AssertionError("Frame-index overlap detected between train and test folds!")
    g = np.asarray(groups)
    overlap = set(g[tr].tolist()) & set(g[te].tolist())
    if overlap:
        raise AssertionError(f"Lineage/group leakage across train and test folds: {overlap}")


@dataclass
class LineageGroupSplitter:
    """Leakage-checked GroupKFold / StratifiedGroupKFold wrapper."""

    n_splits: int = 5
    stratified: bool = False
    random_state: int = 42

    def split(
        self,
        X: np.ndarray,
        y_strata: Optional[np.ndarray],
        groups: np.ndarray,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        g = np.asarray(groups)
        unique_g = np.unique(g)
        if unique_g.size < self.n_splits:
            raise ValueError(
                f"Cannot split {unique_g.size} unique groups into n_splits={self.n_splits}."
            )
        if self.stratified and y_strata is not None:
            splitter = StratifiedGroupKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            iterator = splitter.split(X, y_strata, g)
        else:
            splitter = GroupKFold(n_splits=self.n_splits)
            iterator = splitter.split(X, y_strata, g)

        for tr_idx, te_idx in iterator:
            assert_disjoint_group_split(g, tr_idx, te_idx)
            yield tr_idx, te_idx
