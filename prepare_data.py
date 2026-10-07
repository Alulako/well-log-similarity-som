from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/seg/tutorials-2016/master/1610_Facies_classification/training_data.csv"

# Cada tupla: (segment_id, poço, fácies, profundidade inicial).
# A partir da profundidade inicial são selecionadas 16 amostras consecutivas
# pertencentes à mesma fácies.
SEGMENTS = [
    ("F2_SHRIMPLIN", "SHRIMPLIN", 2, 2911.0),
    ("F2_NEWBY", "NEWBY", 2, 3009.0),
    ("F2_NOLAN", "NOLAN", 2, 2955.5),
    ("F2_CROSS_H_CATTLE", "CROSS H CATTLE", 2, 2633.0),
    ("F3_SHANKLE", "SHANKLE", 3, 2946.0),
    ("F3_SHRIMPLIN", "SHRIMPLIN", 3, 2948.5),
    ("F3_LUKE_G_U", "LUKE G U", 3, 2781.5),
    ("F3_NEWBY", "NEWBY", 3, 2889.0),
    ("F6_NEWBY", "NEWBY", 6, 3024.5),
    ("F6_SHRIMPLIN", "SHRIMPLIN", 6, 2826.5),
    ("F6_SHANKLE", "SHANKLE", 6, 2984.0),
    ("F6_LUKE_G_U", "LUKE G U", 6, 2816.5),
]

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


def extract_segments(rows):
    by_well = {}
    for row in rows:
        by_well.setdefault(row["Well Name"], []).append(row)

    selected = []

    for segment_id, well_name, facies, start_depth in SEGMENTS:
        well_rows = by_well[well_name]

        start_idx = next(
            i
            for i, row in enumerate(well_rows)
            if float(row["Depth"]) == start_depth and int(row["Facies"]) == facies
        )

        segment_rows = well_rows[start_idx : start_idx + 16]

        if len(segment_rows) != 16:
            raise RuntimeError(f"{segment_id}: segmento incompleto.")

        if not all(int(row["Facies"]) == facies for row in segment_rows):
            raise RuntimeError(
                f"{segment_id}: as 16 amostras não pertencem à mesma fácies."
            )

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


if __name__ == "__main__":
    main()
