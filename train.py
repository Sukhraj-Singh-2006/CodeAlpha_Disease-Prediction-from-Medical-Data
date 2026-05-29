import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

df = pd.read_csv("data/HeartDiseaseTrain-Test.csv")

encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
}

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    results[name] = acc

    print(f"{name}: {acc:.4f}")

best_model = models["Random Forest"]

best_model.fit(X_train, y_train)

joblib.dump(best_model, "model/heart_model.pkl")
joblib.dump(encoders, "model/encoders.pkl")

pred = best_model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))

# Confusion Matrix

cm = confusion_matrix(y_test, pred)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

plt.savefig("confusion_matrix.png")
plt.close()

# ROC Curve

prob = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, prob)

roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr)

plt.title("ROC Curve")

plt.xlabel("FPR")
plt.ylabel("TPR")

plt.savefig("roc_curve.png")

plt.close()

# Feature Importance

importance = best_model.feature_importances_

plt.figure(figsize=(8,6))

plt.barh(X.columns, importance)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.close()

# Model Comparison

plt.figure(figsize=(8,5))

plt.bar(results.keys(), results.values())

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.tight_layout()

plt.savefig("model_comparison.png")

plt.close()

print("\nTraining Complete!")