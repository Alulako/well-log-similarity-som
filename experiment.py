from __future__ import annotations

import argparse
import csv
import math
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np


FEATURES = ("GR", "PHIND", "PE")
EXPECTED_SEGMENT_LENGTH = 16
N_NEURONS = 8
SOM_ITERATIONS = 2500
N_RUNS = 30
BENCHMARK_REPETITIONS = 5


def load_segments(csv_path: Path):
    segments = OrderedDict()
    metadata = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["segment_id"]
            current_metadata = {
                "facies": int(row["facies"]),
                "well_name": row["well_name"],
                "formation": row["formation"],
            }

            if sid not in segments:
                segments[sid] = []
                metadata[sid] = current_metadata
            elif metadata[sid] != current_metadata:
                raise ValueError(f"Metadados inconsistentes no segmento {sid}.")

            values = [float(row[name]) for name in FEATURES]
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Valor não finito encontrado no segmento {sid}.")
            segments[sid].append(values)

    arrays = {
        sid: np.asarray(values, dtype=float)
        for sid, values in segments.items()
    }

    if not arrays:
        raise ValueError("Nenhum segmento foi carregado.")

    lengths = {sid: len(values) for sid, values in arrays.items()}
    if set(lengths.values()) != {EXPECTED_SEGMENT_LENGTH}:
        raise ValueError(
            f"Todos os segmentos devem ter {EXPECTED_SEGMENT_LENGTH} amostras: {lengths}"
        )

    return arrays, metadata


def standardize_segments(segments):
    all_values = np.vstack(list(segments.values()))
    mean = all_values.mean(axis=0)
    std = all_values.std(axis=0, ddof=0)
    std[std == 0] = 1.0

    standardized = {
        sid: (values - mean) / std
        for sid, values in segments.items()
    }
    return standardized, mean, std


class OneDimensionalSOM:
    """SOM 1D compacto, implementado com NumPy para manter o experimento simples."""

    def __init__(self, n_neurons: int, input_dim: int, seed: int):
        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.rng = np.random.default_rng(seed)
        self.weights = None
        self.positions = np.arange(n_neurons, dtype=float)

    def fit(self, data: np.ndarray, iterations: int):
        if len(data) < self.n_neurons:
            raise ValueError("O número de amostras deve ser >= ao número de neurônios.")

        init_idx = self.rng.choice(
            len(data),
            size=self.n_neurons,
            replace=False,
        )
        self.weights = data[init_idx].copy()

        initial_lr = 0.45
        final_lr = 0.05
        initial_sigma = max(1.0, self.n_neurons / 3)
        final_sigma = 0.65

        for step in range(iterations):
            x = data[self.rng.integers(0, len(data))]
            progress = step / max(1, iterations - 1)

            lr = initial_lr * (1 - progress) + final_lr * progress
            sigma = initial_sigma * (1 - progress) + final_sigma * progress

            bmu = int(np.argmin(np.linalg.norm(self.weights - x, axis=1)))
            neighborhood = np.exp(
                -((self.positions - bmu) ** 2) / (2 * sigma * sigma)
            )[:, None]
            self.weights += lr * neighborhood * (x - self.weights)

        return self

    def transform(self, sequence: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("O SOM precisa ser treinado antes da transformação.")

        distances = np.linalg.norm(
            sequence[:, None, :] - self.weights[None, :, :],
            axis=2,
        )
        bmu = np.argmin(distances, axis=1).astype(float)

        # A saída é uma codificação escalar pela posição do BMU na cadeia 1D.
        # O treinamento incentiva organização topológica, mas não a garante.
        return bmu / max(1, self.n_neurons - 1)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Custo acumulado de DTW, sem normalização posterior pelo caminho."""
    scalar_mode = a.ndim == 1 and b.ndim == 1

    if not scalar_mode:
        if a.ndim == 1:
            a = a[:, None]
        if b.ndim == 1:
            b = b[:, None]

    n, m = len(a), len(b)
    costs = np.full((n + 1, m + 1), np.inf, dtype=float)
    costs[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if scalar_mode:
                local = abs(float(a[i - 1]) - float(b[j - 1]))
            else:
                local = float(np.linalg.norm(a[i - 1] - b[j - 1]))

            costs[i, j] = local + min(
                costs[i - 1, j],
                costs[i, j - 1],
                costs[i - 1, j - 1],
            )

    return float(costs[n, m])


def distance_matrix(sequences, ids):
    n = len(ids)
    matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(sequences[ids[i]], sequences[ids[j]])
            matrix[i, j] = d
            matrix[j, i] = d

    return matrix


def nearest_neighbor_summary(matrix, ids, metadata):
    total_facies_credit = 0.0
    rows = []

    for i, sid in enumerate(ids):
        distances = matrix[i].copy()
        distances[i] = np.inf
        min_distance = float(np.min(distances))

        tied = np.flatnonzero(
            np.isclose(
                distances,
                min_distance,
                rtol=1e-12,
                atol=1e-12,
            )
        )
        neighbor_ids = tuple(ids[int(j)] for j in tied)
        neighbor_facies = tuple(
            metadata[neighbor]["facies"]
            for neighbor in neighbor_ids
        )

        # Métrica tie-aware: em caso de empate, distribui crédito igualmente
        # entre todos os candidatos mínimos, evitando qualquer viés de ordem/nome.
        facies_credit = float(
            np.mean(
                [
                    facies == metadata[sid]["facies"]
                    for facies in neighbor_facies
                ]
            )
        )
        total_facies_credit += facies_credit

        rows.append(
            {
                "segment_id": sid,
                "facies": metadata[sid]["facies"],
                "neighbor_ids": neighbor_ids,
                "neighbor_facies": neighbor_facies,
                "distance": min_distance,
                "facies_credit": facies_credit,
                "tie_count": int(len(tied)),
            }
        )

    return total_facies_credit / len(ids), rows


def preserved_neighbor_fraction(original_rows, reduced_rows):
    """Acordo esperado sob desempate uniforme entre vizinhos empatados."""
    original = {
        row["segment_id"]: set(row["neighbor_ids"])
        for row in original_rows
    }

    scores = []
    for row in reduced_rows:
        original_set = original[row["segment_id"]]
        reduced_set = set(row["neighbor_ids"])
        intersection = len(original_set & reduced_set)

        # Probabilidade de duas escolhas uniformes (uma de cada conjunto
        # empatado) selecionarem o mesmo vizinho.
        scores.append(
            intersection / (len(original_set) * len(reduced_set))
        )

    return float(np.mean(scores))


def separation_ratio(matrix, ids, metadata):
    within = []
    between = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if metadata[ids[i]]["facies"] == metadata[ids[j]]["facies"]:
                within.append(matrix[i, j])
            else:
                between.append(matrix[i, j])

    mean_within = float(np.mean(within))
    mean_between = float(np.mean(between))
    ratio = mean_between / mean_within if mean_within else math.inf
    return mean_within, mean_between, ratio


def upper_triangle(matrix):
    return np.asarray(
        [
            matrix[i, j]
            for i in range(len(matrix))
            for j in range(i + 1, len(matrix))
        ],
        dtype=float,
    )


def pearson_distance_correlation(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _timed_matrix(sequences, ids):
    start = time.perf_counter()
    distance_matrix(sequences, ids)
    return (time.perf_counter() - start) * 1000.0


def benchmark_pair(original, reduced, ids, seed):
    """Cronometra apenas a etapa de comparação, com representações já disponíveis."""
    original_times = []
    reduced_times = []

    for repetition in range(BENCHMARK_REPETITIONS):
        original_first = (seed + repetition) % 2 == 0

        if original_first:
            original_times.append(_timed_matrix(original, ids))
            reduced_times.append(_timed_matrix(reduced, ids))
        else:
            reduced_times.append(_timed_matrix(reduced, ids))
            original_times.append(_timed_matrix(original, ids))

    return float(np.mean(original_times)), float(np.mean(reduced_times))


def stats(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def matrix_to_csv(matrix, ids):
    lines = ["," + ",".join(ids)]
    for sid, row in zip(ids, matrix):
        lines.append(
            sid + "," + ",".join(f"{value:.6f}" for value in row)
        )
    return "\n".join(lines) + "\n"


def rows_to_csv(rows):
    fieldnames = [
        "segment_id",
        "facies",
        "nearest_neighbors",
        "neighbor_facies",
        "distance",
        "facies_credit",
        "tie_count",
    ]
    out = [",".join(fieldnames)]

    for row in rows:
        out.append(
            ",".join(
                [
                    str(row["segment_id"]),
                    str(row["facies"]),
                    "|".join(row["neighbor_ids"]),
                    "|".join(str(v) for v in row["neighbor_facies"]),
                    f'{row["distance"]:.6f}',
                    f'{row["facies_credit"]:.6f}',
                    str(row["tie_count"]),
                ]
            )
        )

    return "\n".join(out) + "\n"


def run(data_path: Path):
    segments, metadata = load_segments(data_path)
    ids = list(segments.keys())

    standardized, mean, std = standardize_segments(segments)
    all_data = np.vstack([standardized[sid] for sid in ids])

    original_matrix = distance_matrix(standardized, ids)
    original_acc, original_nn = nearest_neighbor_summary(
        original_matrix,
        ids,
        metadata,
    )
    original_within, original_between, original_ratio = separation_ratio(
        original_matrix,
        ids,
        metadata,
    )

    per_run = []
    reduced_matrix_sum = np.zeros_like(original_matrix)

    for seed in range(N_RUNS):
        som = OneDimensionalSOM(
            N_NEURONS,
            len(FEATURES),
            seed=seed,
        )
        som.fit(all_data, iterations=SOM_ITERATIONS)

        reduced = {
            sid: som.transform(standardized[sid])
            for sid in ids
        }

        reduced_matrix = distance_matrix(reduced, ids)
        reduced_matrix_sum += reduced_matrix

        reduced_acc, reduced_nn = nearest_neighbor_summary(
            reduced_matrix,
            ids,
            metadata,
        )
        preserved_nn = preserved_neighbor_fraction(
            original_nn,
            reduced_nn,
        )
        reduced_within, reduced_between, reduced_ratio = separation_ratio(
            reduced_matrix,
            ids,
            metadata,
        )
        pearson_r = pearson_distance_correlation(
            upper_triangle(original_matrix),
            upper_triangle(reduced_matrix),
        )

        original_ms, reduced_ms = benchmark_pair(
            standardized,
            reduced,
            ids,
            seed=seed,
        )

        per_run.append(
            {
                "seed": seed,
                "nn_accuracy": reduced_acc,
                "preserved_nearest_neighbor": preserved_nn,
                "within_distance": reduced_within,
                "between_distance": reduced_between,
                "separation_ratio": reduced_ratio,
                "pearson_distance_correlation": pearson_r,
                "original_runtime_ms": original_ms,
                "reduced_runtime_ms": reduced_ms,
            }
        )

    reduced_matrix_mean = reduced_matrix_sum / N_RUNS
    _, reduced_nn_from_mean_matrix = nearest_neighbor_summary(
        reduced_matrix_mean,
        ids,
        metadata,
    )

    reduced_acc_mean, reduced_acc_std = stats(
        [row["nn_accuracy"] for row in per_run]
    )
    preserved_mean, preserved_std = stats(
        [row["preserved_nearest_neighbor"] for row in per_run]
    )
    within_mean, within_std = stats(
        [row["within_distance"] for row in per_run]
    )
    between_mean, between_std = stats(
        [row["between_distance"] for row in per_run]
    )
    ratio_mean, ratio_std = stats(
        [row["separation_ratio"] for row in per_run]
    )
    pearson_mean, pearson_std = stats(
        [row["pearson_distance_correlation"] for row in per_run]
    )
    original_runtime_mean, original_runtime_std = stats(
        [row["original_runtime_ms"] for row in per_run]
    )
    reduced_runtime_mean, reduced_runtime_std = stats(
        [row["reduced_runtime_ms"] for row in per_run]
    )

    speedup = (
        original_runtime_mean / reduced_runtime_mean
        if reduced_runtime_mean
        else math.inf
    )

    return {
        "ids": ids,
        "metadata": metadata,
        "mean": mean,
        "std": std,
        "original_matrix": original_matrix,
        "original_nn": original_nn,
        "reduced_matrix_mean": reduced_matrix_mean,
        "reduced_nn_from_mean_matrix": reduced_nn_from_mean_matrix,
        "per_run": per_run,
        "metrics": {
            "original_nn_accuracy": original_acc,
            "original_within_distance": original_within,
            "original_between_distance": original_between,
            "original_separation_ratio": original_ratio,
            "reduced_nn_accuracy_mean": reduced_acc_mean,
            "reduced_nn_accuracy_std": reduced_acc_std,
            "preserved_nearest_neighbor_mean": preserved_mean,
            "preserved_nearest_neighbor_std": preserved_std,
            "reduced_within_distance_mean": within_mean,
            "reduced_within_distance_std": within_std,
            "reduced_between_distance_mean": between_mean,
            "reduced_between_distance_std": between_std,
            "reduced_separation_ratio_mean": ratio_mean,
            "reduced_separation_ratio_std": ratio_std,
            "pearson_distance_correlation_mean": pearson_mean,
            "pearson_distance_correlation_std": pearson_std,
            "original_runtime_ms_mean": original_runtime_mean,
            "original_runtime_ms_std": original_runtime_std,
            "reduced_runtime_ms_mean": reduced_runtime_mean,
            "reduced_runtime_ms_std": reduced_runtime_std,
            "speedup_ratio_of_means": speedup,
        },
    }


def summary_markdown(result):
    m = result["metrics"]

    return f"""# Resultado do experimento

## Pergunta testada

Uma representação reduzida baseada em SOM consegue preservar relações úteis de similaridade entre segmentos de perfis de poços e reduzir o custo da etapa de comparação?

## Dados e recorte

Foram usados 12 segmentos reais do conjunto público da SEG: 3 fácies, 4 segmentos por fácies, 16 amostras por segmento e 3 curvas (GR, PHIND e PE).

Os rótulos de fácies foram usados **na seleção do recorte e na avaliação**, mas não entram como atributos nem como alvos do treinamento do SOM. O recorte é estratificado e deliberadamente pequeno. Vizinhos do mesmo poço são permitidos porque o objetivo é estudar a geometria deste conjunto fixo, não estimar desempenho em poços novos.

## Comparações

- Baseline: três curvas padronizadas + DTW multivariado dependente.
- Redução: SOM 1D com {N_NEURONS} neurônios + DTW 1D.
- O SOM é repetido {N_RUNS} vezes, com sementes de 0 a {N_RUNS - 1}.
- O DTW usa **custo acumulado**, sem normalização pelo comprimento do caminho.
- O tempo reportado mede **somente a construção da matriz de distâncias com as representações já disponíveis**; treinamento e transformação do SOM ficam fora do cronômetro.

## Resultados

| Métrica | Original | SOM reduzido — média ± DP |
|---|---:|---:|
| Acurácia do vizinho mais próximo por fácies (empates fracionados) | {m['original_nn_accuracy']:.1%} | {m['reduced_nn_accuracy_mean']:.1%} ± {m['reduced_nn_accuracy_std']:.1%} |
| Acordo com o primeiro vizinho do baseline (empates fracionados) | — | {m['preserved_nearest_neighbor_mean']:.1%} ± {m['preserved_nearest_neighbor_std']:.1%} |
| Razão de separação inter/intrafácies | {m['original_separation_ratio']:.3f} | {m['reduced_separation_ratio_mean']:.3f} ± {m['reduced_separation_ratio_std']:.3f} |
| Correlação de Pearson entre distâncias e baseline | 1,000 | {m['pearson_distance_correlation_mean']:.3f} ± {m['pearson_distance_correlation_std']:.3f} |
| Tempo da matriz de distâncias | {m['original_runtime_ms_mean']:.1f} ± {m['original_runtime_ms_std']:.1f} ms | {m['reduced_runtime_ms_mean']:.1f} ± {m['reduced_runtime_ms_std']:.1f} ms |

Razão entre as médias de tempo (original/reduzido): **{m['speedup_ratio_of_means']:.2f}×**.

## Interpretação

As métricas respondem a perguntas diferentes. A correlação de Pearson mede associação global entre as distâncias das duas representações; a acurácia por fácies mede se o conjunto de vizinhos empatados na menor distância pertence à mesma classe; e o acordo de primeiro vizinho mede a concordância com o baseline. Em empates, o crédito é fracionado uniformemente entre os candidatos mínimos, sem usar nomes ou rótulos para desempatar.

Nenhuma dessas medidas, isoladamente, demonstra preservação completa da similaridade. Este resultado é uma **prova de conceito descritiva em um conjunto fixo**, não uma estimativa de generalização para poços novos nem uma validação geológica ampla.

## Observação sobre agregação

Os resultados principais acima são a **média das métricas calculadas separadamente em cada uma das {N_RUNS} sementes**. O arquivo distance_matrix_reduced_mean.csv contém, separadamente, a média elemento a elemento das {N_RUNS} matrizes reduzidas e serve apenas como visualização agregada; ele não representa uma execução individual do SOM.
"""


def html_report(result):
    m = result["metrics"]
    ids = result["ids"]
    metadata = result["metadata"]

    segments = "".join(
        "<tr>"
        f"<td>{sid}</td>"
        f"<td>{metadata[sid]['well_name']}</td>"
        f"<td>{metadata[sid]['formation']}</td>"
        f"<td>{metadata[sid]['facies']}</td>"
        "</tr>"
        for sid in ids
    )

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Experimento SOM + DTW — Similaridade de perfis</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1050px; margin: 40px auto; padding: 0 20px; line-height: 1.5; color: #202124; }}
h1, h2 {{ margin-bottom: 8px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 20px 0 30px; }}
.card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; background: #fafafa; }}
.card strong {{ display: block; font-size: 13px; margin-bottom: 8px; }}
.card span {{ font-size: 22px; font-weight: 700; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 30px; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: left; }}
.note {{ background: #f4f6f8; border-left: 4px solid #555; padding: 14px; }}
</style>
</head>
<body>
<h1>Preservação de similaridade após redução por SOM</h1>
<p>Prova de conceito com 12 segmentos reais, 16 amostras por segmento, três fácies e três curvas: GR, PHIND e PE.</p>

<div class="grid">
<div class="card"><strong>Acurácia NN — original</strong><span>{m['original_nn_accuracy']:.1%}</span></div>
<div class="card"><strong>Acurácia NN — SOM</strong><span>{m['reduced_nn_accuracy_mean']:.1%} ± {m['reduced_nn_accuracy_std']:.1%}</span></div>
<div class="card"><strong>Mesmo NN preservado</strong><span>{m['preserved_nearest_neighbor_mean']:.1%} ± {m['preserved_nearest_neighbor_std']:.1%}</span></div>
<div class="card"><strong>Pearson das distâncias</strong><span>{m['pearson_distance_correlation_mean']:.3f} ± {m['pearson_distance_correlation_std']:.3f}</span></div>
</div>

<h2>Resultados principais</h2>
<table>
<thead><tr><th>Métrica</th><th>Original</th><th>SOM reduzido</th></tr></thead>
<tbody>
<tr><td>Acurácia do vizinho mais próximo por fácies (empates fracionados)</td><td>{m['original_nn_accuracy']:.3f}</td><td>{m['reduced_nn_accuracy_mean']:.3f} ± {m['reduced_nn_accuracy_std']:.3f}</td></tr>
<tr><td>Acordo com o primeiro vizinho do baseline (empates fracionados)</td><td>—</td><td>{m['preserved_nearest_neighbor_mean']:.3f} ± {m['preserved_nearest_neighbor_std']:.3f}</td></tr>
<tr><td>Razão de separação</td><td>{m['original_separation_ratio']:.3f}</td><td>{m['reduced_separation_ratio_mean']:.3f} ± {m['reduced_separation_ratio_std']:.3f}</td></tr>
<tr><td>Correlação de Pearson com as distâncias originais</td><td>1,000</td><td>{m['pearson_distance_correlation_mean']:.3f} ± {m['pearson_distance_correlation_std']:.3f}</td></tr>
<tr><td>Tempo da matriz de distâncias</td><td>{m['original_runtime_ms_mean']:.1f} ± {m['original_runtime_ms_std']:.1f} ms</td><td>{m['reduced_runtime_ms_mean']:.1f} ± {m['reduced_runtime_ms_std']:.1f} ms</td></tr>
</tbody>
</table>

<div class="note">
O benchmark mede apenas a construção da matriz de distâncias com as representações já disponíveis. O treinamento/transformação pelo SOM não está incluído. Os rótulos de fácies são usados na seleção do recorte e na avaliação, não no treinamento.
</div>

<h2>Segmentos usados</h2>
<table><thead><tr><th>Segmento</th><th>Poço</th><th>Formação</th><th>Fácies</th></tr></thead><tbody>{segments}</tbody></table>
</body>
</html>
"""


def print_summary(result):
    m = result["metrics"]

    print("=== EXPERIMENTO SOM + DTW ===")
    print("12 segmentos reais | 3 facies | 16 amostras/segmento | GR, PHIND, PE")
    print(f"SOM repetido em {N_RUNS} sementes diferentes")
    print()
    print(f"Acuracia NN - original:              {m['original_nn_accuracy']:.3f}")
    print(
        "Acuracia NN - SOM:                   "
        f"{m['reduced_nn_accuracy_mean']:.3f} +/- {m['reduced_nn_accuracy_std']:.3f}"
    )
    print(
        "Mesmo primeiro vizinho preservado:   "
        f"{m['preserved_nearest_neighbor_mean']:.3f} +/- "
        f"{m['preserved_nearest_neighbor_std']:.3f}"
    )
    print(f"Razao separacao - original:          {m['original_separation_ratio']:.3f}")
    print(
        "Razao separacao - SOM:               "
        f"{m['reduced_separation_ratio_mean']:.3f} +/- "
        f"{m['reduced_separation_ratio_std']:.3f}"
    )
    print(
        "Pearson distancias original/reduzido:"
        f" {m['pearson_distance_correlation_mean']:.3f} +/- "
        f"{m['pearson_distance_correlation_std']:.3f}"
    )
    print(
        "Tempo matriz original:               "
        f"{m['original_runtime_ms_mean']:.1f} +/- "
        f"{m['original_runtime_ms_std']:.1f} ms"
    )
    print(
        "Tempo matriz reduzida:                "
        f"{m['reduced_runtime_ms_mean']:.1f} +/- "
        f"{m['reduced_runtime_ms_std']:.1f} ms"
    )
    print(
        "Razao das medias de tempo:            "
        f"{m['speedup_ratio_of_means']:.2f}x"
    )
    print()
    print("Interprete as métricas em conjunto; nenhum valor isolado confirma a hipótese.")


def build_output_files(result):
    ids = result["ids"]
    m = result["metrics"]

    summary_rows = [
        ("nn_accuracy", m["original_nn_accuracy"], m["reduced_nn_accuracy_mean"], m["reduced_nn_accuracy_std"]),
        ("first_neighbor_agreement_tie_aware", "", m["preserved_nearest_neighbor_mean"], m["preserved_nearest_neighbor_std"]),
        ("separation_ratio", m["original_separation_ratio"], m["reduced_separation_ratio_mean"], m["reduced_separation_ratio_std"]),
        ("pearson_distance_correlation", 1.0, m["pearson_distance_correlation_mean"], m["pearson_distance_correlation_std"]),
        ("runtime_ms", m["original_runtime_ms_mean"], m["reduced_runtime_ms_mean"], m["reduced_runtime_ms_std"]),
        ("runtime_original_std_ms", m["original_runtime_ms_std"], "", ""),
        ("speedup_ratio_of_means", m["speedup_ratio_of_means"], "", ""),
    ]

    def fmt(value):
        return f"{value:.6f}" if isinstance(value, (int, float)) else ""

    summary_lines = ["metric,original_value,reduced_mean,reduced_std"]
    for metric, original, reduced_mean, reduced_std in summary_rows:
        summary_lines.append(
            f"{metric},{fmt(original)},{fmt(reduced_mean)},{fmt(reduced_std)}"
        )

    summary_csv = "\n".join(summary_lines) + "\n"

    per_seed_lines = [
        "seed,nn_accuracy_tie_aware,first_neighbor_agreement_tie_aware,within_distance,"
        "between_distance,separation_ratio,pearson_distance_correlation,"
        "original_runtime_ms,reduced_runtime_ms"
    ]
    for row in result["per_run"]:
        per_seed_lines.append(
            f"{row['seed']},{row['nn_accuracy']:.6f},"
            f"{row['preserved_nearest_neighbor']:.6f},"
            f"{row['within_distance']:.6f},{row['between_distance']:.6f},"
            f"{row['separation_ratio']:.6f},"
            f"{row['pearson_distance_correlation']:.6f},"
            f"{row['original_runtime_ms']:.6f},"
            f"{row['reduced_runtime_ms']:.6f}"
        )

    reduced_matrix_mean_csv = matrix_to_csv(
        result["reduced_matrix_mean"],
        ids,
    )
    reduced_nn_mean_csv = rows_to_csv(
        result["reduced_nn_from_mean_matrix"]
    )

    return {
        "metrics_summary.csv": summary_csv,
        "metrics.csv": summary_csv,
        "per_seed_metrics.csv": "\n".join(per_seed_lines) + "\n",
        "distance_matrix_original.csv": matrix_to_csv(
            result["original_matrix"],
            ids,
        ),
        "distance_matrix_reduced_mean.csv": reduced_matrix_mean_csv,
        "nearest_neighbors_original.csv": rows_to_csv(
            result["original_nn"]
        ),
        "nearest_neighbors_reduced_from_mean_matrix.csv": reduced_nn_mean_csv,
        # Aliases legados, explicitamente documentados no README.
        "distance_matrix_reduced.csv": reduced_matrix_mean_csv,
        "nearest_neighbors_reduced.csv": reduced_nn_mean_csv,
        "summary.md": summary_markdown(result),
        "report.html": html_report(result),
    }


def save_outputs(result, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in build_output_files(result).items():
        (output_dir / filename).write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/selected_segments.csv"),
        help="CSV com os 12 segmentos selecionados.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Salva resultados consolidados em results/.",
    )
    args = parser.parse_args()

    result = run(args.data)
    print_summary(result)

    if args.save:
        save_outputs(result, Path("results"))
        print("\nArquivos atualizados em results/.")


if __name__ == "__main__":
    main()
