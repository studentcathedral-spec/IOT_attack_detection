import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------
# 1. Load Dataset
# ---------------------------------------

df = pd.read_csv("RT_IOT2022.csv")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())


# ---------------------------------------
# 2. Check missing values
# ---------------------------------------

print("\nMissing values:")
print(df.isnull().sum())


# Remove rows containing missing values
df = df.dropna()


# ---------------------------------------
# 3. Identify target column
# ---------------------------------------

# RT-IoT2022 uses the 'Attack_type' column
target_column = "Attack_type"

X = df.drop(columns=[target_column])
y = df[target_column]


# ---------------------------------------
# 4. Convert categorical features
# ---------------------------------------

# Convert categorical columns into numerical values
categorical_columns = X.select_dtypes(
    include=["object"]
).columns

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)


# ---------------------------------------
# 5. Encode target labels
# ---------------------------------------

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("\nAttack classes:")
print(label_encoder.classes_)


# ---------------------------------------
# 6. Train/Test Split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ---------------------------------------
# 7. Train Random Forest
# ---------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ---------------------------------------
# 8. Make Predictions
# ---------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------
# 9. Calculate Performance
# ---------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")


# ---------------------------------------
# 10. Detailed Classification Report
# ---------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ---------------------------------------
# 11. Confusion Matrix
# ---------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")

plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()


# ---------------------------------------
# 12. Feature Importance
# ---------------------------------------

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(feature_importance.head(15))


# ---------------------------------------
# 13. Plot Feature Importance
# ---------------------------------------

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance.head(15)["Feature"][::-1],
    feature_importance.head(15)["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features Used for IoT Attack Detection")

plt.tight_layout()
plt.show()