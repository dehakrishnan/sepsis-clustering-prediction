# Tasks for Sepsis Subtype Early-Warning System

## Step 1: Data Loading and Preprocessing

* Load patient data from PhysioNet 2019 dataset
* Calculate average, minimum, and maximum values of vital signs over the first 6 hours of ICU stay
* Handle missing values using imputation (e.g., forward-fill or mean imputation)
* Generate a patient summary table

## Step 2: Clustering and Subtype Identification

* Apply clustering (k-means or Gaussian Mixture) to the patient summary table
* Validate the number of clusters using silhouette scores
* Identify 2 to 4 distinct patient subtypes

## Step 3: Subtype-Specific Model Training

* Train a baseline Random Forest model on the entire patient population to predict sepsis onset 6 hours in advance
* Train specialized Random Forest models for each identified patient subtype
* Compare the aggregate performance of subtype-specific models against the baseline model using AUC scores and statistical significance tests (e.g., DeLong test)

## Step 4: Model Evaluation and Comparison

* Evaluate the performance of each subtype-specific model using AUC scores and statistical significance tests (e.g., DeLong test)
* Compare the performance of subtype-specific models against the baseline model
* Identify the most accurate subtype-specific model(s)

## Step 5: Edge Case Handling

* Handle insufficient signal (e.g., almost all columns are missing) by assigning patients to the "nearest" subtype
* Handle low-prevalence subtypes (e.g., <100 cases) by merging or flagging them as having unreliable subtype-specific models
* Handle data distribution shift by assigning patients to the "nearest" subtype and providing a confidence score for the assignment

## Step 6: Deployment and Maintenance

* Deploy the subtype-specific models in a production-ready environment (e.g., Docker, Kubernetes)
* Monitor and maintain the models to ensure they remain accurate and up-to-date
* Continuously evaluate and improve the models based on new data and feedback
