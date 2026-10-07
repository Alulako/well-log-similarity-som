from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import experiment


DATA_PATH = Path("data/selected_segments.csv")
EXPECTED_DEPTH_STEP = 0.5


def validate_depth_grid():
    grouped = {}

    with DATA_PATH.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            grouped.setdefault(row["segment_id"], []).append(row)

    for segment_id, rows in grouped.items():
        depths = np.asarray([float(row["depth"]) for row in rows], dtype=float)
        steps = np.diff(depths)

        assert len(rows) == experiment.EXPECTED_SEGMENT_LENGTH
        assert np.all(steps > 0)
        assert np.allclose(steps, EXPECTED_DEPTH_STEP)
        assert len({row["facies"] for row in rows}) == 1
        assert len({row["well_name"] for row in rows}) == 1


def validate_dtw_symmetry():
    a = np.asarray([0.0, 0.0, 0.0])
    b = np.asarray([1.0, 1.0, 0.0])

    forward = experiment.dtw_distance(a, b)
    backward = experiment.dtw_distance(b, a)
    assert np.isclose(forward, backward)

    segments, _ = experiment.load_segments(DATA_PATH)
    standardized, _, _ = experiment.standardize_segments(segments)
    ids = list(standardized)

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            x = standardized[ids[i]]
            y = standardized[ids[j]]
            dxy = experiment.dtw_distance(x, y)
            dyx = experiment.dtw_distance(y, x)
            assert np.isclose(dxy, dyx, rtol=1e-12, atol=1e-12)


def validate_reduced_symmetry():
    segments, _ = experiment.load_segments(DATA_PATH)
    standardized, _, _ = experiment.standardize_segments(segments)
    ids = list(standardized)
    all_data = np.vstack([standardized[sid] for sid in ids])

    for seed in (0, 7, 29):
        som = experiment.OneDimensionalSOM(
            experiment.N_NEURONS,
            len(experiment.FEATURES),
            seed=seed,
        )
        som.fit(all_data, iterations=experiment.SOM_ITERATIONS)
        reduced = {
            sid: som.transform(standardized[sid])
            for sid in ids
        }

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                x = reduced[ids[i]]
                y = reduced[ids[j]]
                dxy = experiment.dtw_distance(x, y)
                dyx = experiment.dtw_distance(y, x)
                assert np.isclose(dxy, dyx, rtol=1e-12, atol=1e-12)


def validate_tie_handling_is_order_independent():
    segments, metadata = experiment.load_segments(DATA_PATH)
    standardized, _, _ = experiment.standardize_segments(segments)
    ids = list(standardized)

    matrix = experiment.distance_matrix(standardized, ids)
    accuracy, _ = experiment.nearest_neighbor_summary(
        matrix,
        ids,
        metadata,
    )

    permutation = np.arange(len(ids))[::-1]
    reordered_ids = [ids[i] for i in permutation]
    reordered_matrix = matrix[np.ix_(permutation, permutation)]
    reordered_accuracy, _ = experiment.nearest_neighbor_summary(
        reordered_matrix,
        reordered_ids,
        metadata,
    )

    assert np.isclose(accuracy, reordered_accuracy)


def main():
    validate_depth_grid()
    validate_dtw_symmetry()
    validate_reduced_symmetry()
    validate_tie_handling_is_order_independent()
    print(
        "OK: profundidades regulares, DTW simétrico e métricas sem viés "
        "de ordem nos testes executados."
    )


if __name__ == "__main__":
    main()
