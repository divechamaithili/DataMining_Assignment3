# Machine Learning Notebooks: From Zero to Hero

This repository contains six Jupyter notebooks that explore machine learning workflows, algorithms, and practical capabilities. The notebooks are organized around two themes:

- **End-to-end and capability tours** for PyCaret and AutoGluon, plus GPU-accelerated data science with NVIDIA RAPIDS.
- **Algorithm-focused learning** through a detailed K-means clustering notebook.

Each notebook is designed to be explored cell by cell. Run the setup and data-inspection cells first, then execute the relevant sections in order. Some notebooks may require substantial runtime, optional dependencies, or a GPU.

## Notebooks
### 1. `Copy_of_final_kmeans_zero_to_hero.ipynb` — K-means Clustering: From Zero to Hero

A detailed, algorithm-focused notebook that builds understanding of clustering from mathematical foundations through practical applications.

Topics covered:
- What clusters are and how clustering methods differ
- The K-means objective and why the centroid is the mean
- Implementing Lloyd's K-means algorithm from scratch
- Convergence, computational cost, initialization, and K-means++
- Empty clusters, outliers, and bisecting K-means
- Alternative methods and objectives, including k-medians, k-medoids, fuzzy c-means, Gaussian mixtures, kernel K-means, and mini-batch K-means
- Internal and external cluster-validity metrics
- Choosing the number of clusters and assessing stability
- Common K-means failure cases and comparisons with hierarchical clustering and DBSCAN
- End-to-end customer segmentation, image color quantization, and digit clustering
- Cluster explanation, boundary analysis, persistence, monitoring, embeddings, and multimodal clustering
- Worked textbook-style exercises
https://www.youtube.com/watch?v=q5rv-cfvcXc
**Purpose:** Develop a deeper understanding of how clustering works, when K-means assumptions are useful, and how to evaluate and operationalize clustering results.

### 2. `final_autogluon_capabilities_tour.ipynb` — AutoGluon: A Whirlwind Tour of Capabilities

An example-focused companion to the AutoGluon deep dive. It presents a range of problem types using small, recognizable datasets and emphasizes the input data, core API calls, and resulting outputs.

The tour includes examples such as:
- Binary classification using telecom churn
- Multiclass classification using loan grades
- Regression using house prices
- Additional AutoGluon capabilities and task-specific workflows demonstrated in the notebook
https://youtu.be/KPEAuy6P-JA
**Purpose:** Provide a quick reference for the kinds of problems AutoGluon can address and the basic shape of its workflow.

### 3. `final_autogluon_zero_to_hero.ipynb` — AutoGluon: From Zero to Hero

A progressive introduction to AutoGluon and automated machine learning. The notebook explains how to build predictors with a small amount of code, then examines the results and the choices behind them.

Topics covered include:
- Data inspection and baseline models
- Tabular classification and regression
- Leaderboards and evaluation metrics
- Class imbalance, thresholds, and probability calibration
- Model selection and ensemble behavior
- Time series and multimodal workflows
- Interpreting predictions and considering practical deployment concerns
https://youtu.be/11AG94yXBCc
**Purpose:** Understand AutoGluon's AutoML workflow, how to read its model leaderboard, and how to evaluate its outputs rather than treating automation as a substitute for analysis.


### 4. `final_nvidia_rapids_zero_to_hero.ipynb` — NVIDIA RAPIDS: From Zero to Hero

A GPU data-science masterclass focused on NVIDIA RAPIDS, with examples designed around a Colab T4 environment. It demonstrates how familiar data-science operations can be performed on the GPU and where GPU acceleration is useful.

Topics covered:
- GPU setup and performance benchmarking
- cuDF for GPU DataFrames and `cudf.pandas` acceleration
- Data analysis and visualization, including when to sample or move data to the host
- cuML classification and regression
- GPU preprocessing pipelines
- Clustering and dimensionality reduction, including k-means, DBSCAN, HDBSCAN, PCA, UMAP, and t-SNE
- Nearest neighbors and anomaly scoring
- Cross-validation, model search, and `cuml.accel`
- Scaling techniques, GPU memory, and distributed data workflows
- XGBoost, SHAP, and cuGraph graph analytics
- Model persistence, serving considerations, and monitoring
https://youtu.be/OkvnYeIBWP4
**Purpose:** Learn the RAPIDS ecosystem and understand both the performance opportunities and the practical constraints of GPU-based data science.


### 5. `final_pycaret_capabilities_tour_(1).ipynb` — PyCaret: A Whirlwind Tour of Capabilities

A compact, example-driven tour of the different problem types supported by PyCaret. Rather than focusing on one long project, it demonstrates the inputs, high-level calls, and outputs for individual use cases.

The notebook includes examples across classification, regression, clustering, anomaly detection, time series forecasting, and text-related workflows, along with visualizations and evaluation outputs.
https://youtu.be/j35KWR6Awms
**Purpose:** Quickly see how PyCaret can be applied to different machine learning tasks and recognize the common pattern used across its modules.


### 6. `final_pycaret_zero_to_hero.ipynb` — PyCaret: From Zero to Hero
https://youtu.be/LzFn044dl_U
A progressive introduction to low-code machine learning with PyCaret, using a household spending scenario to demonstrate the machine learning lifecycle.

Topics covered:
- Dataset inspection and a scikit-learn baseline
- PyCaret experiment setup and preprocessing options
- Classification: model comparison, evaluation metrics, plots, tuning, ensembles, calibration, and decision thresholds
- Regression and residual analysis
- Avoiding data leakage
- Model interpretability, including SHAP and feature importance
- Clustering and anomaly detection
- Time series forecasting
- Saving and loading models, API generation, experiment logging, fairness checks, and data-drift monitoring

**Purpose:** Learn how PyCaret organizes common ML tasks into a consistent workflow, while understanding the decisions that still require careful attention.


## Suggested learning order

1. **PyCaret: From Zero to Hero** — begin with a guided end-to-end low-code workflow.
2. **PyCaret: A Whirlwind Tour of Capabilities** — review how the same toolkit applies to different tasks.
3. **AutoGluon: From Zero to Hero** — compare a different AutoML approach and study its model selection workflow.
4. **AutoGluon: A Whirlwind Tour of Capabilities** — use the compact examples as a capability reference.
5. **NVIDIA RAPIDS: From Zero to Hero** — explore GPU-accelerated data processing and machine learning.
6. **K-means Clustering: From Zero to Hero** — study one algorithm in depth, from first principles to real-world use cases.

The notebooks can also be opened independently depending on the topic you want to study.

## How to run

These are Jupyter notebooks (`.ipynb`). You can run them locally with Jupyter or upload them to Google Colab.

1. Clone or download this repository.
2. Open the notebook you want to run.
3. Read the introductory notes and install the dependencies listed in the notebook.
4. Run cells from top to bottom, especially where later cells depend on variables created earlier.
5. Review the printed outputs, charts, and evaluation metrics as you go.

### Environment notes

- **PyCaret and AutoGluon:** Install the package versions and optional dependencies required by the notebook. Some workflows may take longer depending on the selected models and hardware.
- **NVIDIA RAPIDS:** GPU examples require a compatible CUDA environment. The notebook is designed around a Colab T4 workflow; local execution may require a different setup.
- **K-means:** Most core examples can run on a CPU, but runtime depends on dataset size and the algorithms being demonstrated.
- **Reproducibility:** Where a random seed is configured, keep it unchanged when comparing runs. Exact results can still vary with library versions, hardware, and execution environment.


## Repository structure

```text
.
├── Copy_of_final_kmeans_zero_to_hero.ipynb
├── final_autogluon_capabilities_tour.ipynb
├── final_autogluon_zero_to_hero.ipynb
├── final_nvidia_rapids_zero_to_hero.ipynb
├── final_pycaret_capabilities_tour_(1).ipynb
├── final_pycaret_zero_to_hero.ipynb
└── README.md
```

## Learning outcomes

After working through these notebooks, you should have practical exposure to:

- Preparing data and building repeatable ML experiments
- Selecting evaluation metrics that fit the task
- Comparing models and understanding automated model selection
- Applying supervised and unsupervised learning methods
- Interpreting predictions and investigating model errors
- Understanding GPU acceleration and its trade-offs
- Saving, serving, and monitoring machine learning models

## Notes
The notebooks are educational walkthroughs. Their examples and outputs should be interpreted in the context of the datasets and runtime used. Re-run the notebooks in your own environment to verify results before relying on them in another project.
