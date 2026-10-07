from __future__ import annotations

import argparse
import csv
import math
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np


FEATURES = ("GR", "PHIND", "PE")
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
            if sid not in segments:
                segments[sid] = []
                metadata[sid] = {
                    "facies": int(row["facies"]),
                    "well_name": row["well_name"],
                    "formation": row["formation"],
                }
            segments[sid].append([float(row[name]) for name in FEATURES])

    segments = {sid: np.asarray(values, dtype=float) for sid, values in segments.items()}
    return segments, metadata


def standardize_segments(segments):
    all_values = np.vstack(list(segments.values()))
    mean = all_values.mean(axis=0)
    std = all_values.std(axis=0)
    std[std == 0] = 1.0
    standardized = {sid: (values - mean) / std for sid, values in segments.items()}
    return standardized, mean, std


class OneDimensionalSOM:
    """SOM 1D pequeno, implementado apenas com NumPy para manter o experimento simples."""

    def __init__(self, n_neurons: int, input_dim: int, seed: int = 42):
        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.rng = np.random.default_rng(seed)
        self.weights = None
        self.positions = np.arange(n_neurons, dtype=float)

    def fit(self, data: np.ndarray, iterations: int = 2500):
        init_idx = self.rng.choice(len(data), size=self.n_neurons, replace=False)
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
        distances = np.linalg.norm(
            sequence[:, None, :] - self.weights[None, :, :], axis=2
        )
        bmu = np.argmin(distances, axis=1).astype(float)
        # Como o SOM é 1D, a posição do neurônio preserva uma ordenação topológica.
        return bmu / max(1, self.n_neurons - 1)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """DTW normalizado. Usa custo absoluto em 1D e Euclidiano no caso multivariado."""
    scalar_mode = a.ndim == 1 and b.ndim == 1
    if not scalar_mode:
        if a.ndim == 1:
            a = a[:, None]
        if b.ndim == 1:
            b = b[:, None]

    n, m = len(a), len(b)
    costs = np.full((n + 1, m + 1), np.inf, dtype=float)
    steps = np.zeros((n + 1, m + 1), dtype=int)
    costs[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if scalar_mode:
                local = abs(float(a[i - 1]) - float(b[j - 1]))
            else:
                local = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            candidates = (
                (costs[i - 1, j], steps[i - 1, j]),
                (costs[i, j - 1], steps[i, j - 1]),
                (costs[i - 1, j - 1], steps[i - 1, j - 1]),
            )
            prev_cost, prev_steps = min(candidates, key=lambda item: item[0])
            costs[i, j] = local + prev_cost
            steps[i, j] = prev_steps + 1

    return costs[n, m] / max(1, steps[n, m])


def distance_matrix(sequences, ids):
    n = len(ids)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(sequences[ids[i]], sequences[ids[j]])
            matrix[i, j] = d
            matrix[j, i] = d
    return matrix


def nearest_neighbor_accuracy(matrix, ids, metadata):
    hits = 0
    rows = []

    for i, sid in enumerate(ids):
        candidates = [(matrix[i, j], ids[j]) for j in range(len(ids)) if j != i]
        distance, neighbor = min(candidates)
        same = metadata[sid]["facies"] == metadata[neighbor]["facies"]
        hits += int(same)
        rows.append(
            {
                "segment_id": sid,
                "facies": metadata[sid]["facies"],
                "nearest_neighbor": neighbor,
                "neighbor_facies": metadata[neighbor]["facies"],
                "distance": distance,
                "same_facies": same,
            }
        )

    return hits / len(ids), rows


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
    values = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            values.append(matrix[i, j])
    return np.asarray(values, dtype=float)


def correlation(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def benchmark(sequences, ids, repetitions=40):
    start = time.perf_counter()
    for _ in range(repetitions):
        distance_matrix(sequences, ids)
    elapsed = time.perf_counter() - start
    return (elapsed / repetitions) * 1000.0


def matrix_to_csv(matrix, ids):
    lines = ["," + ",".join(ids)]
    for sid, row in zip(ids, matrix):
        lines.append(sid + "," + ",".join(f"{value:.6f}" for value in row))
    return "\n".join(lines) + "\n"


def rows_to_csv(rows):
    fieldnames = [
        "segment_id",
        "facies",
        "nearest_neighbor",
        "neighbor_facies",
        "distance",
        "same_facies",
    ]
    out = [",".join(fieldnames)]
    for row in rows:
        out.append(
            ",".join(
                [
                    str(row["segment_id"]),
                    str(row["facies"]),
                    str(row["nearest_neighbor"]),
                    str(row["neighbor_facies"]),
                    f'{row["distance"]:.6f}',
                    str(row["same_facies"]),
                ]
            )
        )
    return "\n".join(out) + "\n"


def stats(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def html_report(result):
    m = result["metrics"]
    ids = result["ids"]
    metadata = result["metadata"]

    segments = "".join(
        "<tr>"
        f"<td>{sid}</td><td>{metadata[sid]['well_name']}</td>"
        f"<td>{metadata[sid]['formation']}</td><td>{metadata[sid]['facies']}</td>"
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
<h1>Experimento pequeno: preservação de similaridade após redução por SOM</h1>
<p>Dados reais de perfilagem do conjunto público de fácies da SEG (2016). Foram usados 12 segmentos, 16 amostras por segmento, três fácies e três curvas: GR, PHIND e PE.</p>

<div class="grid">
<div class="card"><strong>Acurácia NN — original</strong><span>{m['original_nn_accuracy']:.1%}</span></div>
<div class="card"><strong>Acurácia NN — SOM (30 execuções)</strong><span>{m['reduced_nn_accuracy_mean']:.1%} ± {m['reduced_nn_accuracy_std']:.1%}</span></div>
<div class="card"><strong>Correlação das distâncias</strong><span>{m['distance_correlation_mean']:.3f} ± {m['distance_correlation_std']:.3f}</span></div>
<div class="card"><strong>Speedup aproximado</strong><span>{m['speedup']:.2f}×</span></div>
</div>

<h2>Por que 30 execuções?</h2>
<p>O treinamento do SOM envolve inicialização e amostragem aleatórias. Por isso, o experimento é repetido com 30 sementes diferentes e os resultados do espaço reduzido são apresentados como média ± desvio-padrão. Isso evita escolher uma execução excepcionalmente favorável.</p>

<h2>Resultados principais</h2>
<table>
<thead><tr><th>Métrica</th><th>Original</th><th>SOM reduzido</th></tr></thead>
<tbody>
<tr><td>Acurácia do vizinho mais próximo por fácies</td><td>{m['original_nn_accuracy']:.3f}</td><td>{m['reduced_nn_accuracy_mean']:.3f} ± {m['reduced_nn_accuracy_std']:.3f}</td></tr>
<tr><td>Razão de separação</td><td>{m['original_separation_ratio']:.3f}</td><td>{m['reduced_separation_ratio_mean']:.3f} ± {m['reduced_separation_ratio_std']:.3f}</td></tr>
<tr><td>Correlação com a matriz original</td><td>1,000</td><td>{m['distance_correlation_mean']:.3f} ± {m['distance_correlation_std']:.3f}</td></tr>
<tr><td>Tempo da matriz de distâncias</td><td>{m['original_runtime_ms']:.1f} ms</td><td>{m['reduced_runtime_ms_mean']:.1f} ± {m['reduced_runtime_ms_std']:.1f} ms</td></tr>
</tbody>
</table>

<h2>Interpretação</h2>
<p>A redução por SOM preservou parcialmente a estrutura global de distâncias: a correlação média com a matriz original foi positiva e relativamente alta. A razão de separação entre fácies aumentou no espaço reduzido, mas a acurácia do vizinho mais próximo apresentou variabilidade entre execuções e, em média, não superou o baseline original. Portanto, o resultado não sustenta a afirmação de que o SOM melhora consistentemente a identificação do vizinho mais próximo neste recorte.</p>

<div class="note">Conclusão segura: o experimento oferece evidência de que a representação reduzida pode preservar parte da estrutura de similaridade e diminuir o custo da comparação, mas a qualidade de vizinhança depende da inicialização do SOM. É uma prova de conceito, não uma validação geológica geral.</div>

<h2>Segmentos usados</h2>
<table><thead><tr><th>Segmento</th><th>Poço</th><th>Formação</th><th>Fácies</th></tr></thead><tbody>{segments}</tbody></table>
</body>
</html>
"""


def run(data_path: Path):
    segments, metadata = load_segments(data_path)
    ids = list(segments.keys())

    standardized, mean, std = standardize_segments(segments)
    all_data = np.vstack([standardized[sid] for sid in ids])

    original_matrix = distance_matrix(standardized, ids)
    original_acc, original_nn = nearest_neighbor_accuracy(original_matrix, ids, metadata)
    ow, ob, oratio = separation_ratio(original_matrix, ids, metadata)
    original_ms = benchmark(
        standardized, ids, repetitions=BENCHMARK_REPETITIONS * 2
    )

    per_run = []
    reduced_matrix_sum = np.zeros_like(original_matrix)

    for seed in range(N_RUNS):
        som = OneDimensionalSOM(N_NEURONS, len(FEATURES), seed=seed)
        som.fit(all_data, iterations=SOM_ITERATIONS)
        reduced = {sid: som.transform(standardized[sid]) for sid in ids}

        reduced_matrix = distance_matrix(reduced, ids)
        reduced_matrix_sum += reduced_matrix
        reduced_acc, _ = nearest_neighbor_accuracy(reduced_matrix, ids, metadata)
        rw, rb, rratio = separation_ratio(reduced_matrix, ids, metadata)
        corr = correlation(
            upper_triangle(original_matrix), upper_triangle(reduced_matrix)
        )
        reduced_ms = benchmark(
            reduced, ids, repetitions=BENCHMARK_REPETITIONS
        )

        per_run.append(
            {
                "seed": seed,
                "nn_accuracy": reduced_acc,
                "within_distance": rw,
                "between_distance": rb,
                "separation_ratio": rratio,
                "distance_correlation": corr,
                "runtime_ms": reduced_ms,
            }
        )

    acc_mean, acc_std = stats([r["nn_accuracy"] for r in per_run])
    within_mean, within_std = stats([r["within_distance"] for r in per_run])
    between_mean, between_std = stats([r["between_distance"] for r in per_run])
    ratio_mean, ratio_std = stats([r["separation_ratio"] for r in per_run])
    corr_mean, corr_std = stats([r["distance_correlation"] for r in per_run])
    runtime_mean, runtime_std = stats([r["runtime_ms"] for r in per_run])

    speedup = original_ms / runtime_mean if runtime_mean else math.inf
    reduced_matrix_mean = reduced_matrix_sum / N_RUNS
    _, reduced_nn_mean = nearest_neighbor_accuracy(
        reduced_matrix_mean, ids, metadata
    )

    return {
        "ids": ids,
        "metadata": metadata,
        "mean": mean,
        "std": std,
        "original_matrix": original_matrix,
        "original_nn": original_nn,
        "reduced_matrix_mean": reduced_matrix_mean,
        "reduced_nn_mean": reduced_nn_mean,
        "per_run": per_run,
        "metrics": {
            "original_nn_accuracy": original_acc,
            "original_within_distance": ow,
            "original_between_distance": ob,
            "original_separation_ratio": oratio,
            "original_runtime_ms": original_ms,
            "reduced_nn_accuracy_mean": acc_mean,
            "reduced_nn_accuracy_std": acc_std,
            "reduced_within_distance_mean": within_mean,
            "reduced_within_distance_std": within_std,
            "reduced_between_distance_mean": between_mean,
            "reduced_between_distance_std": between_std,
            "reduced_separation_ratio_mean": ratio_mean,
            "reduced_separation_ratio_std": ratio_std,
            "distance_correlation_mean": corr_mean,
            "distance_correlation_std": corr_std,
            "reduced_runtime_ms_mean": runtime_mean,
            "reduced_runtime_ms_std": runtime_std,
            "speedup": speedup,
        },
    }


def print_summary(result):
    m = result["metrics"]

    print("=== EXPERIMENTO SOM + DTW ===")
    print("12 segmentos reais | 3 facies | 16 amostras/segmento | GR, PHIND, PE")
    print(f"SOM repetido em {N_RUNS} sementes diferentes")
    print()
    print(f"Acuracia NN - original (mDTW):       {m['original_nn_accuracy']:.3f}")
    print(
        "Acuracia NN - SOM 1D + DTW:        "
        f"{m['reduced_nn_accuracy_mean']:.3f} +/- {m['reduced_nn_accuracy_std']:.3f}"
    )
    print(f"Razao separacao - original:          {m['original_separation_ratio']:.3f}")
    print(
        "Razao separacao - SOM:             "
        f"{m['reduced_separation_ratio_mean']:.3f} +/- "
        f"{m['reduced_separation_ratio_std']:.3f}"
    )
    print(
        "Correlacao distancias orig/reduz:  "
        f"{m['distance_correlation_mean']:.3f} +/- "
        f"{m['distance_correlation_std']:.3f}"
    )
    print(f"Tempo matriz original:               {m['original_runtime_ms']:.3f} ms")
    print(
        "Tempo matriz reduzida:             "
        f"{m['reduced_runtime_ms_mean']:.3f} +/- "
        f"{m['reduced_runtime_ms_std']:.3f} ms"
    )
    print(f"Speedup aproximado:                  {m['speedup']:.2f}x")
    print()
    print(
        "Conclusao: o SOM preserva parcialmente a estrutura de similaridade e "
        "reduz o custo da comparacao, mas o resultado de vizinho mais proximo "
        "varia entre inicializacoes."
    )


def save_outputs(result, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = result["ids"]
    m = result["metrics"]

    summary_rows = [
        ("nn_accuracy_original", m["original_nn_accuracy"], "", ""),
        (
            "nn_accuracy_reduced",
            "",
            m["reduced_nn_accuracy_mean"],
            m["reduced_nn_accuracy_std"],
        ),
        ("separation_ratio_original", m["original_separation_ratio"], "", ""),
        (
            "separation_ratio_reduced",
            "",
            m["reduced_separation_ratio_mean"],
            m["reduced_separation_ratio_std"],
        ),
        (
            "distance_correlation_reduced",
            "",
            m["distance_correlation_mean"],
            m["distance_correlation_std"],
        ),
        ("runtime_original_ms", m["original_runtime_ms"], "", ""),
        (
            "runtime_reduced_ms",
            "",
            m["reduced_runtime_ms_mean"],
            m["reduced_runtime_ms_std"],
        ),
        ("speedup", m["speedup"], "", ""),
    ]

    summary_lines = ["metric,original_value,reduced_mean,reduced_std"]
    for metric, original, reduced_mean, reduced_std in summary_rows:
        original_text = f"{original:.6f}" if isinstance(original, (int, float)) else ""
        mean_text = (
            f"{reduced_mean:.6f}"
            if isinstance(reduced_mean, (int, float))
            else ""
        )
        std_text = (
            f"{reduced_std:.6f}"
            if isinstance(reduced_std, (int, float))
            else ""
        )
        summary_lines.append(
            f"{metric},{original_text},{mean_text},{std_text}"
        )

    (output_dir / "metrics_summary.csv").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    per_seed_lines = [
        "seed,nn_accuracy,within_distance,between_distance,"
        "separation_ratio,distance_correlation,runtime_ms"
    ]
    for row in result["per_run"]:
        per_seed_lines.append(
            f"{row['seed']},{row['nn_accuracy']:.6f},"
            f"{row['within_distance']:.6f},{row['between_distance']:.6f},"
            f"{row['separation_ratio']:.6f},"
            f"{row['distance_correlation']:.6f},{row['runtime_ms']:.6f}"
        )

    (output_dir / "per_seed_metrics.csv").write_text(
        "\n".join(per_seed_lines) + "\n", encoding="utf-8"
    )

    (output_dir / "distance_matrix_original.csv").write_text(
        matrix_to_csv(result["original_matrix"], ids), encoding="utf-8"
    )
    (output_dir / "distance_matrix_reduced.csv").write_text(
        matrix_to_csv(result["reduced_matrix_mean"], ids), encoding="utf-8"
    )
    (output_dir / "nearest_neighbors_original.csv").write_text(
        rows_to_csv(result["original_nn"]), encoding="utf-8"
    )
    (output_dir / "nearest_neighbors_reduced.csv").write_text(
        rows_to_csv(result["reduced_nn_mean"]), encoding="utf-8"
    )

    # Mantém metrics.csv como alias do resumo para evitar resultados antigos.
    (output_dir / "metrics.csv").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    (output_dir / "report.html").write_text(
        html_report(result), encoding="utf-8"
    )


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
        help="Salva tabelas e report.html em results/.",
    )
    args = parser.parse_args()

    result = run(args.data)
    print_summary(result)

    if args.save:
        save_outputs(result, Path("results"))
        print("\nArquivos salvos em results/.")


if __name__ == "__main__":
    main()

