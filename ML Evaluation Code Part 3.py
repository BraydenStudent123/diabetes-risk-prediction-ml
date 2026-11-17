# %% 1. LOAD AND CLEAN DATA
# Checkpoint only: no test-set predictions or plots.
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score
)
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
# Formatting for compact Word report figures
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300
})
from sklearn.model_selection import learning_curve
from sklearn.base import BaseEstimator, ClassifierMixin


file_path = Path(__file__).resolve().parent / "diabetes_012_health_indicators_BRFSS2015.xlsx"
df = pd.read_excel(file_path)
df["CholCheck"] = pd.to_numeric(df["CholCheck"], errors="coerce")
print("Missing values before cleaning:", df.isna().sum().sum())
df = df.dropna().copy()
print("Cleaned dataset shape:", df.shape)
print("Original three-class distribution (%):")
print((df["Diabetes_012"].value_counts(normalize=True).sort_index() * 100).round(2))

# %% 2. FEATURES, LABELS, AND SPLITS
# Binary target: 0 = reported healthy; 1 = reported prediabetes OR diabetes. Define Features 
X = df.drop(columns="Diabetes_012")
y = (df["Diabetes_012"] != 0).astype(int)

# Reserve 20% for later. Do not use it anywhere in this checkpoint. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Same 50,000 development observations for all four models. Development subset; and cross-validation folds; f2 scorer for imbalanced classification. Use the same 5-fold CV for all models. Use the same development subset for all models. Use the same f2 scorer for all models.
X_dev, _, y_dev, _ = train_test_split(
    X_train, y_train, train_size=50000, stratify=y_train, random_state=42
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f2 = make_scorer(fbeta_score, beta=2, zero_division=0)
print("\nTraining size:", len(X_train), "Reserved test size:", len(X_test))
print("Development subset size:", len(X_dev))
print("Binary development distribution (%):")
print((y_dev.value_counts(normalize=True).sort_index() * 100).round(2))

# %% 3. BASELINE AND THREE TUNED COURSE MODELS
baseline = DummyClassifier(strategy="most_frequent")
baseline_scores = cross_val_score(
    baseline, X_dev, y_dev, cv=cv, scoring=f2
)
# Dummy Classifier
results = [{
    "Model": "Dummy (majority class)",
    "Mean CV F2": baseline_scores.mean(),
    "CV F2 Std": baseline_scores.std(),
    "Best parameters": "most_frequent"
}]
#Logistic Regression and hyperparameter grid, Decision Tree and hyperparameter grid, KNN and hyperparameter grid. Use a pipeline for models that require scaling. Use GridSearchCV to find the best hyperparameters for each model. Store the results in a list of dictionaries for later comparison.   
models = {
    "Logistic Regression": (
        Pipeline([("scaler", StandardScaler()),
                  ("model", LogisticRegression(max_iter=1000))]),
        {"model__C": [0.001, 0.005, 0.01, 0.05, 0.1, 1],

"model__class_weight": [
    None,
    "balanced",
    {0: 1, 1: 2},
    {0: 1, 1: 3},
    {0: 1, 1: 4},
    {0: 1, 1: 5},
    {0: 1, 1: 6},
    {0: 1, 1: 7},
{0: 1, 1: 8},
{0: 1, 1: 10},
{0: 1, 1: 12}
]}
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [3, 5, 10, 15],
         "class_weight": [None, "balanced"]}
    ),
    "KNN": (
    Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),
    {"model__n_neighbors": [1, 2, 3, 4, 5, 7, 10, 15, 30]}
)
}

#GridSearchCV for each model, fit on development data, and store results. Use the same cross-validation folds and f2 scorer for all models. Print the best parameters and mean CV F2 score for each model. Store the results in a list of dictionaries for later comparison.
for name, (model, parameter_grid) in models.items():
    print("\nTuning:", name, flush=True)

    search = GridSearchCV(
        model,
        parameter_grid,
        cv=cv,
        scoring=f2,
        n_jobs=-1
    )

    search.fit(X_dev, y_dev)

    results.append({
        "Model": name,
        "Mean CV F2": search.best_score_,
        "CV F2 Std": search.cv_results_["std_test_score"][search.best_index_],
        "Best parameters": str(search.best_params_)
    })


# %% 4. CHECKPOINT MODEL COMPARISON (CROSS-VALIDATION ONLY)
comparison = pd.DataFrame(results).sort_values("Mean CV F2", ascending=False)
print("\n--- FIVE-FOLD CROSS-VALIDATION COMPARISON ---")
print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nBest model based on development CV:", comparison.iloc[0]["Model"])
print("No predictions or evaluation were performed on the reserved test set.")


# %% 5. EXPERIMENTAL THRESHOLD TUNING (DEVELOPMENT DATA ONLY)

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, fbeta_score

# Use the best Logistic Regression settings from your CV results; Selected Logistic Regression model
from sklearn.base import clone

threshold_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
       C=0.05,
class_weight={0: 1, 1: 7},
        max_iter=1000
    )
)])

# Each development observation is predicted by a model
# that did not train on that observation. Out of fold probabilities are used to evaluate thresholds. Threshold testing
dev_probabilities = cross_val_predict(
    threshold_model,
    X_dev,
    y_dev,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

#threshold tuning: Evaluate multiple thresholds and store results in a list of dictionaries. Use the same development data and out-of-fold probabilities for all thresholds. Print the results in a table format. Select the threshold with the highest F2 score for further evaluation.
threshold_results = []

for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:

    predictions = (dev_probabilities >= threshold).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_dev, predictions),
        "Precision": precision_score(y_dev, predictions, zero_division=0),
        "Recall": recall_score(y_dev, predictions, zero_division=0),
        "F1": f1_score(y_dev, predictions, zero_division=0),
        "F2": fbeta_score(y_dev, predictions, beta=2, zero_division=0)
    })

threshold_table = pd.DataFrame(threshold_results)

print("\n--- DEVELOPMENT THRESHOLD COMPARISON ---")
print(threshold_table.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}"
))

best_row = threshold_table.loc[threshold_table["F2"].idxmax()]

print("\nBest F2 threshold:", best_row["Threshold"])
print("Best F2:", round(best_row["F2"], 4))


# Automatically select the threshold with the highest development F2
selected_threshold = float(best_row["Threshold"])

selected_row = threshold_table.loc[
    threshold_table["Threshold"] == selected_threshold
].iloc[0]

print(f"\n--- SELECTED THRESHOLD: {selected_threshold:.2f} ---")
print(selected_row.to_string())


selected_predictions = (
    dev_probabilities >= selected_threshold
).astype(int)

print("\nSelected threshold:", selected_threshold)
print("Precision:", precision_score(y_dev, selected_predictions))
print("Recall:", recall_score(y_dev, selected_predictions))
print("F2:", fbeta_score(y_dev, selected_predictions, beta=2))
print("Percentage flagged:", selected_predictions.mean() * 100)



# %% 6. DIAGNOSIS: LEARNING CURVES

# Custom classifier to apply the selected threshold of 0.50; learning curves for both the 50,000 development observations and the full training portion (~203,000). Use the same cross-validation folds and f2 scorer for all models. Plot learning curves for both datasets. Print numerical results in a table format.
class ThresholdClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, estimator, threshold=0.45):
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        probabilities = self.estimator_.predict_proba(X)[:, 1]
        return (probabilities >= self.threshold).astype(int)


# Selected Logistic Regression model
learning_model = ThresholdClassifier(
    estimator=threshold_model,
    threshold=selected_threshold
)

# Reuse existing F2 scorer
f2_threshold = f2

# Compare both datasets
datasets = {
    "50,000 Development Observations": (X_dev, y_dev),
    "Full Training Portion (~203,000)": (X_train, y_train)
}

sizes = np.linspace(0.1, 1.0, 8)

for dataset_name, (X_data, y_data) in datasets.items():

    train_sizes, train_scores, validation_scores = learning_curve(
        learning_model,
        X_data,
        y_data,
        train_sizes=sizes,
        cv=cv,
        scoring=f2_threshold,
        n_jobs=-1,
        shuffle=True,
        random_state=42
    )

    train_mean = train_scores.mean(axis=1)
    val_mean = validation_scores.mean(axis=1)

    # Plot learning curve
    plt.figure(figsize=(9, 6))

    plt.plot(train_sizes, train_mean, "o-", label="Training F2")
    plt.plot(train_sizes, val_mean, "o-", label="Validation F2")

    plt.xlabel("Training Set Size")
    plt.ylabel("F2 Score")

    plt.title(
        f"Learning Curve: {dataset_name}\n"
        f"Logistic Regression (Threshold = {selected_threshold:.2f})"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Numerical results
    diagnosis = pd.DataFrame({
        "Training Size": train_sizes,
        "Training F2": train_mean,
        "Validation F2": val_mean,
        "Gap": train_mean - val_mean
    })

    print(f"\n--- {dataset_name} ---")
    print(diagnosis.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    ))


# %% 7. COST-RATIO SENSITIVITY ANALYSIS
from sklearn.metrics import confusion_matrix

# Assumed incremental costs per classification; Cost Assumptions;
COST_TP = 145
COST_FP = 145
COST_TN = 0

cost_ratios = {
    "Base (3.45:1)": 500,  #FN cost = 500 ; FP, TP, and TN costs are fixed 
    "5:1": 725,
    "10:1": 1450,
    "20:1": 2900
}

# Test thresholds from 0.01 to 0.99
thresholds = np.arange(0.01, 1.00, 0.01)

cost_results = []
cost_curves = {}

for ratio, cost_fn in cost_ratios.items():

    total_costs = []

    for threshold in thresholds:
        predictions = (dev_probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_dev, predictions, labels=[0, 1]
        ).ravel()

        total_cost = (
            COST_TP * tp +
            COST_FP * fp +
            cost_fn * fn +
            COST_TN * tn
        )

        total_costs.append(total_cost)

    cost_curves[ratio] = total_costs

    best_index = np.argmin(total_costs)
    best_threshold = thresholds[best_index]
    best_cost = total_costs[best_index]

    # Baseline strategies
    n_positive = (y_dev == 1).sum()
    n_negative = (y_dev == 0).sum()

    flag_everyone = (
        n_positive * COST_TP +
        n_negative * COST_FP
    )

    flag_no_one = (
        n_positive * cost_fn +
        n_negative * COST_TN
    )

    cost_results.append({
        "Cost Ratio": ratio,
        "FN Cost": cost_fn,
        "Best Threshold": round(best_threshold, 2),
        "Model Cost": best_cost,
        "Flag Everyone": flag_everyone,
        "Flag No One": flag_no_one,
        "Savings vs Everyone": flag_everyone - best_cost,
        "Savings vs No One": flag_no_one - best_cost
    })

cost_table = pd.DataFrame(cost_results)

print("\n--- COST-RATIO SENSITIVITY ANALYSIS ---")
print(cost_table.to_string(index=False))

# Plot cost versus threshold
plt.figure(figsize=(9, 5))

for ratio, costs in cost_curves.items():
    plt.plot(thresholds, costs, label=f"Cost Ratio {ratio}")

plt.xlabel("Classification Threshold")
plt.ylabel("Estimated Total Cost ($)")
plt.title("Cost-Ratio Sensitivity Analysis")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



# %% 8. FINAL TEST-SET EVALUATION
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

# Both thresholds selected BEFORE test evaluation

base_cost_threshold = float(
    cost_table.loc[
        cost_table["Cost Ratio"] == "Base (3.45:1)",
        "Best Threshold"
    ].iloc[0]
)

selected_thresholds = {
    "F2-optimal": float(best_row["Threshold"]),
    "Cost-optimal": base_cost_threshold
}


# Original cost assumptions
COST_TP = 145
COST_FP = 145
COST_FN = 500
COST_TN = 0

# Fit final model ONCE using full training data ; Final model fit and test probabilities
final_model = clone(threshold_model)
final_model.fit(X_train, y_train)

# Generate test scores ONCE
test_probabilities = final_model.predict_proba(X_test)[:, 1]

# ROC-AUC is independent of classification  ; Confusion matrices, metrics, and costs are calculated for each threshold
auc = roc_auc_score(y_test, test_probabilities)

evaluation_results = []

# Compare both preselected thresholds
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for i, (name, threshold) in enumerate(
    selected_thresholds.items()
):
    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    total_cost = (
        COST_TP * tp +
        COST_FP * fp +
        COST_FN * fn +
        COST_TN * tn
    )

    evaluation_results.append({
        "Method": name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "F2": fbeta_score(
            y_test, predictions, beta=2, zero_division=0
        ),
        "ROC-AUC": auc,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Total Cost": total_cost
    })

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=[0, 1],
        display_labels=["Negative", "Positive"],
        cmap="Blues",
        values_format="d",
        ax=axes[i],
        colorbar=False
    )
    axes[i].set_title(f"{name} (Threshold = {threshold:.2f})")

plt.tight_layout()
plt.show()

# Print final comparison
evaluation_table = pd.DataFrame(evaluation_results)

print("\n--- FINAL TEST-SET COMPARISON ---")
print(evaluation_table.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}"
))

# ROC curve (same for both thresholds)
fpr, tpr, _ = roc_curve(y_test, test_probabilities)

plt.figure(figsize=(7, 5))
plt.plot(
    fpr, tpr,
    label=f"Logistic Regression (AUC = {auc:.3f})"
)
plt.plot([0, 1], [0, 1], "k--", label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Final Test ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Cost against threshold (test evaluation only)
test_thresholds = np.arange(0.01, 1.00, 0.01)
test_costs = []

for threshold in test_thresholds:
    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    cost = (
        COST_TP * tp +
        COST_FP * fp +
        COST_FN * fn +
        COST_TN * tn
    )

    test_costs.append(cost)

plt.figure(figsize=(9, 5))
plt.plot(test_thresholds, test_costs, label="Test Cost")

# Highlight the thresholds selected using development data

plt.axvline(
    selected_threshold, color="green", linestyle="--",
    label=f"F2-optimal ({selected_threshold:.2f})"
)
plt.axvline(
    base_cost_threshold, color="red", linestyle="--",
    label=f"Cost-optimal ({base_cost_threshold:.2f})"
)


plt.axvline(
    0.50, color="gray", linestyle=":",
    label="Default (0.50)"
)

plt.xlabel("Decision Threshold")
plt.ylabel("Estimated Total Cost ($)")
plt.title("Test Cost vs. Decision Threshold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Compare default, low, high, and selected thresholds
print("\n--- TEST COST AT SELECTED THRESHOLDS ---")

for threshold in sorted(set([0.20, selected_threshold, 0.50, base_cost_threshold, 0.90])):
    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    cost = (
        COST_TP * tp +
        COST_FP * fp +
        COST_FN * fn +
        COST_TN * tn
    )

    print(
        f"Threshold {threshold:.2f}: "
        f"Estimated cost = ${cost:,.0f}"
    )










# %% 9. PRESENTATION DEMONSTRATION
# Uses development data only. Does not touch the reserved test set.
RUN_PRESENTATION_DEMO = True

if RUN_PRESENTATION_DEMO:

    from pathlib import Path
    from sklearn.model_selection import cross_validate, cross_val_predict
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, fbeta_score, confusion_matrix,
        ConfusionMatrixDisplay, roc_auc_score
    )

    presentation_folder = Path("presentation_figures")
    presentation_folder.mkdir(exist_ok=True)

    # ==================================================
    # 1. LIVE DEMONSTRATION CONTROLS
    # ==================================================

    DEMO_C = 0.05
    DEMO_CLASS_WEIGHT = 7
    DEMO_K = 1
    DEMO_DEPTH = 5

    DEMO_THRESHOLD = 0.50

    # Costs from your written report
    DEMO_COST_TP = 145
    DEMO_COST_FP = 145
    DEMO_COST_FN = 500
    DEMO_COST_TN = 0

    # ==================================================
    # 2. DEFINE ALL THREE MODELS
    # ==================================================

    demo_models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=DEMO_C,
                class_weight={0: 1, 1: DEMO_CLASS_WEIGHT},
                max_iter=1000
            ))
        ]),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=DEMO_K
            ))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=DEMO_DEPTH,
            class_weight="balanced",
            random_state=42
        )
    }

    # ==================================================
    # 3. EVALUATE EACH MODEL
    # ==================================================

    demo_summary = []

    threshold_values = np.arange(0.05, 0.96, 0.05)

    for model_name, model in demo_models.items():

        print("\n" + "=" * 65)
        print("MODEL:", model_name)
        print("=" * 65)

        # Five-fold CV using default threshold 0.50
        cv_result = cross_validate(
            model,
            X_dev,
            y_dev,
            cv=cv,
            scoring=f2,
            n_jobs=-1
        )

        cv_scores = cv_result["test_score"]

        print(f"Mean CV F2: {cv_scores.mean():.4f}")
        print(f"CV F2 Std:  {cv_scores.std():.4f}")

        # Out-of-fold probabilities
        probabilities = cross_val_predict(
            model,
            X_dev,
            y_dev,
            cv=cv,
            method="predict_proba",
            n_jobs=-1
        )[:, 1]

        predictions = (
            probabilities >= DEMO_THRESHOLD
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_dev, predictions, labels=[0, 1]
        ).ravel()

        accuracy = accuracy_score(y_dev, predictions)
        precision = precision_score(
            y_dev, predictions, zero_division=0
        )
        recall = recall_score(
            y_dev, predictions, zero_division=0
        )
        f1 = f1_score(
            y_dev, predictions, zero_division=0
        )
        f2_score = fbeta_score(
            y_dev, predictions, beta=2, zero_division=0
        )

        total_cost = (
            DEMO_COST_TP * tp +
            DEMO_COST_FP * fp +
            DEMO_COST_FN * fn +
            DEMO_COST_TN * tn
        )

        print(f"\nThreshold: {DEMO_THRESHOLD:.2f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"F2:        {f2_score:.4f}")
        print(f"ROC-AUC:   {roc_auc_score(y_dev, probabilities):.4f}")
        print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"Estimated cost: ${total_cost:,.0f}")

        demo_summary.append({
            "Model": model_name,
            "Threshold": DEMO_THRESHOLD,
            "Mean CV F2": cv_scores.mean(),
            "CV F2 Std": cv_scores.std(),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "F2": f2_score,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Estimated Cost": total_cost
        })

        # ==================================================
        # 4. CONFUSION MATRIX
        # ==================================================

        fig, ax = plt.subplots(figsize=(6, 5))

        ConfusionMatrixDisplay.from_predictions(
            y_dev,
            predictions,
            labels=[0, 1],
            display_labels=["Healthy", "Abnormal"],
            cmap="Blues",
            values_format="d",
            colorbar=False,
            ax=ax
        )

        ax.set_title(
            f"{model_name}\n"
            f"Confusion Matrix (Threshold = {DEMO_THRESHOLD:.2f})"
        )

        plt.tight_layout()

        filename = model_name.lower().replace(" ", "_")

        plt.savefig(
            presentation_folder / f"{filename}_confusion_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()

        # ==================================================
        # 5. THRESHOLD AND COST CALCULATIONS
        # ==================================================

        threshold_rows = []

        for threshold in threshold_values:

            threshold_predictions = (
                probabilities >= threshold
            ).astype(int)

            t_n, f_p, f_n, t_p = confusion_matrix(
                y_dev, threshold_predictions, labels=[0, 1]
            ).ravel()

            cost = (
                DEMO_COST_TP * t_p +
                DEMO_COST_FP * f_p +
                DEMO_COST_FN * f_n +
                DEMO_COST_TN * t_n
            )

            threshold_rows.append({
                "Threshold": threshold,
                "Accuracy": accuracy_score(
                    y_dev, threshold_predictions
                ),
                "Precision": precision_score(
                    y_dev, threshold_predictions, zero_division=0
                ),
                "Recall": recall_score(
                    y_dev, threshold_predictions, zero_division=0
                ),
                "F2": fbeta_score(
                    y_dev, threshold_predictions,
                    beta=2, zero_division=0
                ),
                "Cost": cost
            })

        threshold_df = pd.DataFrame(threshold_rows)

        # ==================================================
        # 6. ACCURACY, PRECISION, RECALL, F2 VS THRESHOLD
        # ==================================================

        fig, ax = plt.subplots(figsize=(9, 5))

        for metric in ["Accuracy", "Precision", "Recall", "F2"]:
            ax.plot(
                threshold_df["Threshold"],
                threshold_df[metric],
                marker="o",
                label=metric
            )

        ax.axvline(
            DEMO_THRESHOLD,
            color="black",
            linestyle="--",
            label=f"Selected threshold = {DEMO_THRESHOLD:.2f}"
        )

        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_title(f"{model_name}: Threshold Performance")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            presentation_folder / f"{filename}_threshold_metrics.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()

        # ==================================================
        # 7. COST VS DECISION THRESHOLD
        # ==================================================

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(
            threshold_df["Threshold"],
            threshold_df["Cost"],
            marker="o",
            label="Estimated cost"
        )

        ax.axvline(
            DEMO_THRESHOLD,
            color="black",
            linestyle="--",
            label=f"Selected = {DEMO_THRESHOLD:.2f}"
        )

        # Exact selected-threshold cost
        ax.scatter(
            [DEMO_THRESHOLD],
            [total_cost],
            color="red",
            s=80,
            zorder=5,
            label=f"Selected cost = ${total_cost:,.0f}"
        )

        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Estimated Cost ($)")
        ax.set_title(f"{model_name}: Cost vs Decision Threshold")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            presentation_folder / f"{filename}_cost_curve.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.show()

        threshold_df.to_csv(
            presentation_folder / f"{filename}_threshold_table.csv",
            index=False
        )

    # ==================================================
    # 8. ALL-MODEL COMPARISON
    # ==================================================

    demo_table = pd.DataFrame(demo_summary)

    print("\n" + "=" * 65)
    print("FINAL PRESENTATION COMPARISON")
    print("=" * 65)

    print(
        demo_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    demo_table.to_csv(
        presentation_folder / "demo_model_comparison.csv",
        index=False
    )

    print("\nAll presentation figures saved to:")
    print(presentation_folder.resolve())

# %%
