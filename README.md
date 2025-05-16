# MovieLens Recommendation System

This project implements and compares two collaborative filtering models (Neural Collaborative Filtering and Autoencoder-based Collaborative Filtering) on the MovieLens 1M dataset using PyTorch.

## Table of Contents
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Setup](#setup)
- [Running Experiments](#running-experiments)
- [Code Overview](#code-overview)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

---

## Project Structure
```
CS412_A3/
├── data/
│   └── movielens_1m/
│       └── ratings.dat
├── results/
│   └── results.txt
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── train.py
│   └── models/
│       ├── ncf_mlp.py
│       └── autoencoder.py
├── requirements.txt
└── run_experiment.bat
```

## Requirements
- Python 3.8+
- PyTorch
- pandas
- scikit-learn

Install dependencies with:
```sh
pip install -r requirements.txt
```

## Dataset
- Download the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/).
- Place the `ratings.dat` file in `data/movielens_1m/ratings.dat`.

## Setup
1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Ensure the dataset is in place:**
   - The file `data/movielens_1m/ratings.dat` must exist.

## Running Experiments
You can run the experiment using:
```sh
python src/evaluate.py
```
Or, on Windows, double-click or run:
```sh
run_experiment.bat
```

This will:
- Train both the NCF_MLP and Autoencoder models on the MovieLens 1M data.
- Evaluate their performance (MAE) on the test set.
- Save results to `results/results.txt`.

## Code Overview
- **src/data_loader.py**: Loads and preprocesses the MovieLens data, mapping user and movie IDs to contiguous indices.
- **src/models/ncf_mlp.py**: Defines the Neural Collaborative Filtering (MLP) model.
- **src/models/autoencoder.py**: Defines the Autoencoder-based collaborative filtering model.
- **src/train.py**: Contains training loops for both models.
- **src/evaluate.py**: Orchestrates training, evaluation, and saving results.
- **results/results.txt**: Output file with model MAE scores.

## Results
After running, you will find a file at `results/results.txt` with content like:
```
Model	MAE
NCF_MLP	0.XXXX
Autoencoder	0.XXXX
```

## Troubleshooting
- **IndexError: index out of range in self**: Ensure you have the correct `ratings.dat` file and that the code is mapping user/movie IDs to contiguous indices (already handled in this repo).
- **FileNotFoundError**: Make sure the dataset is in the correct location and the `results` directory exists (the code will create it if missing).
- **CUDA not available**: The code will automatically use CPU if CUDA is not available.
