from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path

SOURCE_COMMIT = "4885188ff29684ca2774662231f3adad6c3ec563"
SOURCE_URL = (
    "https://raw.githubusercontent.com/seg/tutorials-2016/"
    f"{SOURCE_COMMIT}/1610_Facies_classification/training_data.csv"
)

EXPECTED_SAMPLES = 16
EXPECTED_DEPTH_STEP = 0.5

# Recorte pragmático para uma prova de conceito pequena.
# Cada tupla: (segment_id, poço, fácies, profundidade inicial).
# As fácies foram usadas deliberadamente para montar três grupos balanceados.
SEGMENTS_FILE = Path("data/segment_indices.csv")


def load_segments():
    """Lê os índices de seleção; os valores de perfilagem vêm da fonte original."""
    with SEGMENTS_FILE.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 12:
        raise ValueError("O manifesto deve conter 12 segmentos.")

    segments = []
    for row in rows:
        start = float(row["start_depth_ft"])
        end = float(row["end_depth_ft"])
        samples = int(row["n_samples"])
        if samples != EXPECTED_SAMPLES or abs(end - (start + (samples - 1) * EXPECTED_DEPTH_STEP)) > 1e-9:
            raise ValueError(f"Intervalo inconsistente: {row['segment_id']}")
        segments.append((row["segment_id"], row["well_name"], int(row["facies"]), start))
    return segments


OUTPUT_COLUMNS = [
    "segment_id",
    "facies",
    "formation",
    "well_name",
    "depth",
    "GR",
    "PHIND",
    "PE",
]


def download_rows():
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _check_regular_depth(segment_id, rows):
    depths = [float(row["Depth"]) for row in rows]

    if any(b <= a for a, b in zip(depths, depths[1:])):
        raise RuntimeError(f"{segment_id}: profundidades não são estritamente crescentes.")

    for a, b in zip(depths, depths[1:]):
        step = b - a
        if abs(step - EXPECTED_DEPTH_STEP) > 1e-9:
            raise RuntimeError(
                f"{segment_id}: espaçamento irregular ({a} -> {b}, passo {step})."
            )


def extract_segments(rows):
    by_well = {}
    for row in rows:
        by_well.setdefault(row["Well Name"], []).append(row)

    selected = []

    for segment_id, well_name, facies, start_depth in load_segments():
        well_rows = by_well[well_name]

        start_idx = next(
            i
            for i, row in enumerate(well_rows)
            if float(row["Depth"]) == start_depth and int(row["Facies"]) == facies
        )

        segment_rows = well_rows[start_idx : start_idx + EXPECTED_SAMPLES]

        if len(segment_rows) != EXPECTED_SAMPLES:
            raise RuntimeError(f"{segment_id}: segmento incompleto.")

        if not all(int(row["Facies"]) == facies for row in segment_rows):
            raise RuntimeError(
                f"{segment_id}: as {EXPECTED_SAMPLES} amostras não pertencem à mesma fácies."
            )

        _check_regular_depth(segment_id, segment_rows)

        for row in segment_rows:
            selected.append(
                {
                    "segment_id": segment_id,
                    "facies": facies,
                    "formation": row["Formation"],
                    "well_name": well_name,
                    "depth": row["Depth"],
                    "GR": row["GR"],
                    "PHIND": row["PHIND"],
                    "PE": row["PE"],
                }
            )

    return selected


def main():
    rows = download_rows()
    selected = extract_segments(rows)

    output = Path("data/selected_segments.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(selected)

    print(f"{len(selected)} linhas salvas em {output}.")
    print("12 segmentos x 16 amostras, distribuídos em 3 fácies.")
    print(f"Fonte fixada no commit SEG: {SOURCE_COMMIT}")


if __name__ == "__main__":
    main()
