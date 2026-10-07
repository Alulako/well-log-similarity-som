# Well Log Similarity with SOM

Small, reproducible proof of concept for evaluating whether a SOM-based reduction preserves useful similarity relationships between short well-log segments.

## Research question

Can a compact 1D representation obtained with a Self-Organizing Map preserve enough of the similarity structure of multivariate well-log segments to support simpler comparisons?

The experiment is intentionally small. It is not intended as a geological validation study or as a reproduction of the experiments from the research group.

## Dataset

The data come from the public facies-classification dataset released by the Society of Exploration Geophysicists (SEG) and associated with:

Hall, B. (2016). *Facies classification using machine learning*. The Leading Edge, 35(10), 906–909. DOI: 10.1190/tle35100906.1.

The source file is pinned to commit:

4885188ff29684ca2774662231f3adad6c3ec563

Source file:

https://raw.githubusercontent.com/seg/tutorials-2016/4885188ff29684ca2774662231f3adad6c3ec563/1610_Facies_classification/training_data.csv

The selected subset contains:

- 12 real segments;
- 16 samples per segment;
- 3 facies: 2, 3 and 6;
- 4 segments per facies;
- 6 distinct wells;
- 3 curves: GR, PHIND and PE;
- a regular depth step of 0.5 within each segment.

### How the subset was chosen

The facies labels were used to build a small, balanced subset with three groups and four homogeneous segments per group. They are therefore used both in **subset selection** and later in **evaluation**, but never as SOM input features or training targets.

The starting depths were chosen pragmatically after inspecting the available homogeneous runs in the public CSV. This was not a preregistered sampling rule. The goal was to obtain a compact exercise with equal-length, regularly sampled segments.

GR, PHIND and PE were selected because they are numeric curves available without missing values in the selected rows and keep the input dimensionality small enough for a classroom experiment.

Neighbors from the same well are allowed. The experiment describes the geometry of this fixed subset; it does not estimate performance on unseen wells.

## Experimental design

The same standardized data are represented in two ways:

1. **Original representation** — GR, PHIND and PE compared with dependent multivariate DTW.
2. **Reduced representation** — each 3D sample is quantized by a 1D SOM with 8 neurons; the resulting scalar sequence is compared with 1D DTW.

The SOM uses:

- 8 neurons;
- 2,500 updates;
- 30 independent initializations, using seeds 0–29.

The choices of 8 neurons and 2,500 updates are fixed pragmatic settings for this proof of concept. They were not optimized against the reported evaluation metrics.

The SOM implementation uses only NumPy. It is a simple SOM-based encoding and is **not** an implementation of SOrS/IntraSOM.

## DTW definition

DTW uses cumulative alignment cost:

- absolute local cost for the 1D representation;
- Euclidean local cost for the multivariate representation;
- no post-hoc division by warping-path length.

All compared segments contain 16 samples.

A validation script checks the symmetry of the DTW implementation, the regular depth grid of the selected data, and that tie-aware nearest-neighbor scoring is independent of input order.

## Metrics

The metrics deliberately measure different aspects of preservation:

- **Nearest-neighbor facies accuracy**: whether the closest segment belongs to the same facies. If multiple segments tie at the minimum distance, credit is divided equally among the tied candidates.
- **First-neighbor agreement**: whether the first neighbor from the original representation remains among the minimum-distance neighbors after reduction. Ties are handled by uniform fractional credit rather than by identifier order.
- **Separation ratio**: mean inter-facies distance divided by mean intra-facies distance.
- **Pearson correlation of pairwise distances**: linear association between the 66 pairwise distances in the original and reduced spaces.
- **Comparison runtime**: time to build the pairwise distance matrix once both representations already exist.

The Pearson coefficient is not interpreted as a percentage of preserved similarity. Tie handling is independent of segment names and facies labels.

## Repeated SOM runs

Because SOM training is stochastic, the reduced representation is evaluated across 30 seeds. Reduced-space metrics are reported as **mean ± sample standard deviation**.

This variation reflects SOM initialization/sampling on the same fixed dataset. It is not a confidence interval for generalization to other wells.

## Runtime measurement

Runtime comparisons use the same number of repetitions for both representations. Their execution order is alternated to reduce ordering effects.

The benchmark covers **only the distance-matrix construction step**. SOM training and transformation are excluded, so the runtime result should be interpreted only as the cost of comparison after both representations are available.

## Results

Current results are generated automatically by the experiment and saved in:

- results/summary.md
- results/metrics_summary.csv
- results/per_seed_metrics.csv
- results/report.html

This README intentionally does not duplicate the numerical results so it cannot become stale after a methodological change.

Two aggregate files deserve special attention:

- results/distance_matrix_reduced_mean.csv is the element-wise mean of the 30 reduced distance matrices.
- results/nearest_neighbors_reduced_from_mean_matrix.csv is computed from that mean matrix.

They are descriptive aggregates and do not represent a single SOM execution. The files distance_matrix_reduced.csv and nearest_neighbors_reduced.csv are kept only as legacy aliases of those aggregates.

## Reproducing the experiment

The reference environment used during revision was:

- Python 3.12.4
- NumPy 2.4.6

Install the pinned dependency:

~~~bash
pip install -r requirements.txt
~~~

Run validation checks:

~~~bash
python validate_experiment.py
~~~

Run the experiment:

~~~bash
python experiment.py
~~~

Generate/update all result files:

~~~bash
python experiment.py --save
~~~

Rebuild the selected subset from the pinned public source:

~~~bash
python prepare_data.py
~~~

On Windows, rodar_experimento.bat runs the experiment and updates the result files.

## Repository structure

~~~text
.
├── data/
│   └── selected_segments.csv
├── experiment.py
├── prepare_data.py
├── validate_experiment.py
├── requirements.txt
├── environment.txt
├── rodar_experimento.bat
└── results/
    ├── summary.md
    ├── metrics_summary.csv
    ├── per_seed_metrics.csv
    ├── distance_matrix_original.csv
    ├── distance_matrix_reduced_mean.csv
    ├── nearest_neighbors_original.csv
    ├── nearest_neighbors_reduced_from_mean_matrix.csv
    └── report.html
~~~

## Scope and limitations

This is a descriptive proof of concept on a deliberately selected fixed subset. It does not use a train/test split because the question is about the geometry of this particular dataset, not predictive generalization. The conclusions should not be generalized to other basins, wells or logging configurations without additional experiments.

## References

- HALL, B. Facies classification using machine learning. *The Leading Edge*, 35(10), 906–909, 2016. DOI: 10.1190/tle35100906.1.
- KOHONEN, T. *Self-Organizing Maps*. Springer, 3rd ed., 2001.
- SAKOE, H.; CHIBA, S. Dynamic programming algorithm optimization for spoken word recognition. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 26(1), 43–49, 1978. DOI: 10.1109/TASSP.1978.1163055.
