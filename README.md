# diabetes-risk-prediction-ml
Machine learning classification of diabetes risk using logistic regression, decision trees, and KNN, with cost-sensitive evaluation.
Diabetes Risk Prediction Using Machine Learning

Project Overview
This project investigates the use of supervised machine learning algorithms to classify diabetes-related health outcomes using the CDC Diabetes Health Indicators dataset.
The objective is to evaluate predictive performance while accounting for the asymmetric costs of false negatives and false positives in health screening.

Methods
-Three supervised learning models were evaluated:
-Logistic Regression
-Decision Tree
-K-Nearest Neighbors (KNN)

The project includes:
-Data preprocessing and stratified sampling
-Hyperparameter tuning using 5-fold cross-validation
-Classification threshold optimization
-Precision, recall, F2-score, and ROC-AUC evaluation
-Cost-sensitive decision analysis

Results
-Logistic Regression demonstrated strong predictive performance, with an ROC-AUC of approximately 0.82.
-The project also examined how changing the classification threshold affects sensitivity, false alarms, and estimated screening costs.

Technologies
-Python, scikit-learn, pandas, NumPy, Matplotlib

Dataset
-CDC Diabetes Health Indicators, available through the UCI Machine Learning Repository.

Future Improvements
-Evaluate additional classification algorithms
-Explore model calibration and interpretability
-Investigate robustness across patient subgroups
-Develop a reproducible prediction pipeline

Academic Context:
-Developed as part of graduate coursework in Applied Machine Learning at Vanderbilt University; AI such as ChatGPT was permitted in the making by the instructor.
