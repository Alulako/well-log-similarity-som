# Well Log Similarity with SOM

Small, reproducible experiment for evaluating whether a SOM-based dimensionality reduction preserves useful similarity relationships between well-log segments while reducing comparison cost.

## Dataset

The experiment uses a public well-log facies dataset from the Society of Exploration Geophysicists (SEG), associated with:

Hall, B. (2016). *Facies classification using machine learning*. The Leading Edge, 35(10), 906–909. DOI: 10.1190/tle35100906.1.

Source:
https://github.com/seg/tutorials-2016/tree/master/1610_Facies_classification

Original CSV:
https://raw.githubusercontent.com/seg/tutorials-2016/master/1610_Facies_classification/training_data.csv

To keep the experiment small and easy to reproduce, the selected subset contains:

- 12 real well-log segments;
- 16 samples per segment;
- 3 facies: 2, 3 and 6;
- 4 segments per facies;
- 3 curves: GR, PHIND and PE.

Facies labels are not used to train the SOM. They are used only as an external reference for evaluation.

## Experimental design

Two representations of the same 12 segments are compared:

1. **Original representation**: standardized GR, PHIND and PE curves compared with multivariate DTW.
2. **Reduced representation**: each 3D sample is mapped to a 1D SOM with 8 neurons, and the resulting sequence is compared with 1D DTW.

Because SOM training involves random initialization and sampling, the reduced experiment is repeated **30 times using seeds 0–29**. Results are reported as **mean ± standard deviation**.

The SOM implementation uses only NumPy to keep the code compact and dependency-light. It is a simple SOM-based representation and is not an implementation of SOrS/IntraSOM.

## Metrics

The experiment reports:

- **Nearest-neighbor facies accuracy**: whether the nearest segment belongs to the same facies.
- **Separation ratio**: mean inter-facies distance divided by mean intra-facies distance.
- **Distance-matrix correlation**: correlation between original and reduced pairwise distances.
- **Runtime**: time required to build the pairwise distance matrix.

## Results

| Metric | Original (3D + mDTW) | Reduced (SOM 1D + DTW) |
|---|---:|---:|
| Nearest-neighbor facies accuracy | 58.3% | 48.6% ± 8.2% |
| Separation ratio | 1.367 | 2.543 ± 0.360 |
| Correlation with original distance matrix | 1.000 | 0.758 ± 0.062 |

Runtime depends on the machine and system load. In the local verification run, the reduced comparison was roughly **1.5× faster**.

## Interpretation

The reduced representation preserved part of the global similarity structure, with a mean distance-matrix correlation of 0.758, and increased the separation ratio between facies.

However, nearest-neighbor accuracy decreased on average and varied across SOM initializations. Therefore, the experiment does **not** support the claim that SOM consistently improves nearest-neighbor identification in this subset.

The safer conclusion is that SOM-based reduction can preserve part of the similarity structure and reduce comparison cost, but it may introduce variability and information loss depending on initialization.

This is a small proof of concept and should not be interpreted as a general geological validation.

## Reproducing the experiment

Requirements:

- Python 3
- NumPy

Install dependencies:

~~~bash
pip install -r requirements.txt
~~~

Run the experiment:

~~~bash
python experiment.py
~~~

Run and save result files:

~~~bash
python experiment.py --save
~~~

Rebuild the selected subset from the public source:

~~~bash
python prepare_data.py
~~~

On Windows, you can also run:

~~~text
rodar_experimento.bat
~~~

## Repository structure

~~~text
.
├── data/
│   └── selected_segments.csv
├── experiment.py
├── prepare_data.py
├── requirements.txt
├── rodar_experimento.bat
└── results/
    ├── metrics_summary.csv
    ├── per_seed_metrics.csv
    ├── distance_matrix_original.csv
    ├── distance_matrix_reduced.csv
    ├── nearest_neighbors_original.csv
    ├── nearest_neighbors_reduced.csv
    ├── report.html
    └── summary.md
~~~

## References

- HALL, B. Facies classification using machine learning. *The Leading Edge*, 35(10), 906–909, 2016. DOI: 10.1190/tle35100906.1.
- KOHONEN, T. *Self-Organizing Maps*. Springer, 3rd ed., 2001.
- SAKOE, H.; CHIBA, S. Dynamic programming algorithm optimization for spoken word recognition. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 26(1), 43–49, 1978. DOI: 10.1109/TASSP.1978.1163055.
