# ============================================================
# Machine Learning-Based Detection of IoT Network Attacks
# Dataset: RT-IoT2022
# Algorithms: Decision Tree and Random Forest
# ============================================================

import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# 1. Load dataset
# Put RT_IOT2022.csv in the same folder as this script.
file_path = "RT_IOT2022.csv"
df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)

# Remove index-like columns, e.g. Unnamed: 0
df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed(:|$)", case=False)]
df.columns = df.columns.astype(str).str.strip()

target_column = "Attack_type"
if target_column not in df.columns:
    raise ValueError(
        f"Target column '{target_column}' was not found. "
        f"Check the CSV filename and column names."
    )

# Drop rows with missing values
df = df.dropna().copy()
print("Cleaned dataset shape:", df.shape)
print("Unnamed columns remaining:",
      [c for c in df.columns if str(c).lower().startswith("unnamed")])

# 2. Graph: attack-type distribution
attack_counts = df[target_column].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=attack_counts.index.astype(str), y=attack_counts.values)
plt.title("Distribution of Attack Types in RT-IoT2022")
plt.xlabel("Attack Type")
plt.ylabel("Number of Records")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.savefig("graph_1_attack_type_distribution.png", dpi=300)
plt.show()

print("\nAttack-type counts:\n", attack_counts.to_string())

# 3. Prepare features and target
X = df.drop(columns=[target_column])
y = df[target_column].astype(str)

# Convert categorical input columns to numeric dummy columns
X = pd.get_dummies(X, drop_first=True)

# Encode attack labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nNumber of encoded features:", X.shape[1])
print("Attack classes:", list(label_encoder.classes_))

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 5. Define two algorithms for a fair comparison
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}
predictions = {}

# 6. Train and evaluate both models
for model_name, model in models.items():
    print("\n" + "=" * 60)
    print("Model:", model_name)
    print("=" * 60)

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted",
                                zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted",
                          zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted",
                           zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro",
                        zero_division=0)

    results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Weighted Precision": precision,
        "Weighted Recall": recall,
        "Weighted F1-score": weighted_f1,
        "Macro F1-score": macro_f1,
        "Training Time (seconds)": training_time
    })

    trained_models[model_name] = model
    predictions[model_name] = y_pred

    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Weighted precision: {precision:.4f}")
    print(f"Weighted recall: {recall:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        labels=list(range(len(label_encoder.classes_))),
        target_names=label_encoder.classes_,
        zero_division=0
    ))

# 7. Comparison table and CSV output
comparison_df = pd.DataFrame(results)
print("\nALGORITHM COMPARISON")
print(comparison_df.to_string(
    index=False,
    formatters={
        "Accuracy": "{:.4f}".format,
        "Weighted Precision": "{:.4f}".format,
        "Weighted Recall": "{:.4f}".format,
        "Weighted F1-score": "{:.4f}".format,
        "Macro F1-score": "{:.4f}".format,
        "Training Time (seconds)": "{:.2f}".format
    }
))
comparison_df.to_csv("algorithm_comparison.csv", index=False)
print("\nSaved: algorithm_comparison.csv")

# 8. Graph: accuracy comparison
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=comparison_df, x="Algorithm", y="Accuracy")
ax.set_ylim(0, 1.05)
plt.title("Accuracy Comparison: Decision Tree vs Random Forest")
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
for container in ax.containers:
    ax.bar_label(container, fmt="%.4f", padding=3)
plt.tight_layout()
plt.savefig("graph_2_accuracy_comparison.png", dpi=300)
plt.show()

# 9. Graph: all evaluation metrics
metrics_df = comparison_df.melt(
    id_vars="Algorithm",
    value_vars=[
        "Accuracy", "Weighted Precision", "Weighted Recall",
        "Weighted F1-score", "Macro F1-score"
    ],
    var_name="Metric",
    value_name="Score"
)
plt.figure(figsize=(11, 6))
sns.barplot(data=metrics_df, x="Metric", y="Score", hue="Algorithm")
plt.ylim(0, 1.05)
plt.title("Evaluation Metrics by Algorithm")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.xticks(rotation=20, ha="right")
plt.legend(title="Algorithm")
plt.tight_layout()
plt.savefig("graph_3_metrics_comparison.png", dpi=300)
plt.show()

# 10. Graph: Random Forest confusion matrix
rf_pred = predictions["Random Forest"]
labels = list(range(len(label_encoder.classes_)))
cm = confusion_matrix(y_test, rf_pred, labels=labels)

# Annotate only if the matrix is small enough to remain readable
fig_size = max(10, min(20, len(label_encoder.classes_) * 0.55))
plt.figure(figsize=(fig_size, fig_size * 0.8))
sns.heatmap(
    cm,
    annot=(len(label_encoder.classes_) <= 15),
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Attack Type")
plt.ylabel("Actual Attack Type")
plt.xticks(rotation=60, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("graph_4_random_forest_confusion_matrix.png", dpi=300)
plt.show()

# 11. Graph: top 15 Random Forest feature importances
rf_model = trained_models["Random Forest"]
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

top_features = feature_importance.head(15).sort_values("Importance")
plt.figure(figsize=(10, 7))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.title("Top 15 Important Features - Random Forest")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("graph_5_random_forest_feature_importance.png", dpi=300)
plt.show()

feature_importance.to_csv("random_forest_feature_importance.csv", index=False)
print("Saved: random_forest_feature_importance.csv")

# 12. Simple summary
best_model = comparison_df.loc[comparison_df["Macro F1-score"].idxmax()]
print("\nBest model by Macro F1-score:", best_model["Algorithm"])
print(f"Macro F1-score: {best_model['Macro F1-score']:.4f}")

print("\nSaved graph files:")
print("graph_1_attack_type_distribution.png")
print("graph_2_accuracy_comparison.png")
print("graph_3_metrics_comparison.png")
print("graph_4_random_forest_confusion_matrix.png")
print("graph_5_random_forest_feature_importance.png")

print(
    "\nResearch note: RT-IoT2022 may have imbalanced attack classes. "
    "Do not rely on accuracy alone; review per-class precision, recall, "
    "F1-score and Macro F1-score before drawing conclusions."
)
