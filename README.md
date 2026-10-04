# MLOps Assignment 1: Version Control with Git and GitHub

**Student ID:** 24i-8024

A small house price prediction project used to practise a Git and GitHub workflow for ML work. Source code, data and model artifacts live in separate folders, and only code and configuration files are versioned.

## Project structure

```
├── data/              # raw dataset (dataset.csv), ignored by Git
├── src/
│   └── train.py
├── model/             # trained model output, ignored by Git
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Run these from the project root.

```bash
git clone https://github.com/TauseefTausee/24i-8024-Git_MLOps_Assignment01.git
cd 24i-8024-Git_MLOps_Assignment01

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

## Train the model

```bash
python src/train.py
```

The script loads `data/dataset.csv`, trains a Gradient Boosting regressor and saves it to `model/model_24i-8024.pkl`.

The dataset is not stored in this repository. If `data/dataset.csv` is missing, the script creates a sample house price dataset first, so the command above works on a fresh clone.
