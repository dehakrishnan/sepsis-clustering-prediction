# Technical Architecture Plan

## Directory Layout

* `data/`: contains the PhysioNet 2019 dataset
* `src/`: contains the source code for the project
* `docs/`: contains the documentation for the project
* `results/`: contains the output files from the project

## 6-Hour Window Preprocessing Modules

* `window_preprocessor.py`: responsible for loading the patient data, calculating average, minimum, and maximum values of vital signs over the first 6 hours of ICU stay, and handling missing values
* `missingness_feature_extractor.py`: responsible for extracting the "missingness" feature for each patient to capture the frequency of missing data points

## Random Forest/Clustering Setup

* `random_forest.py`: responsible for training a baseline Random Forest model on the entire patient population to predict sepsis onset 6 hours in advance
* `clustering.py`: responsible for applying clustering (k-means or Gaussian Mixture) to identify 2 to 4 distinct patient subtypes
* `subtype_specific_models.py`: responsible for training specialized Random Forest models for each identified patient subtype

## Technical Requirements

* Python 3.8 or higher
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn

## Development Environment

* Jupyter Notebook
* Visual Studio Code
* PyCharm

## Testing Environment

* Pytest
* Coverage.py

## Deployment Environment

* Docker
* Kubernetes

