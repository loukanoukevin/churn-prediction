"""
Étape 6-7 : interprétation du modèle final et export des figures.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

X_train = pd.read_csv("../data/X_train.csv")
X_test = pd.read_csv("../data/X_test.csv")
y_train = pd.read_csv("../data/y_train.csv").squeeze()
y_test = pd.read_csv("../data/y_test.csv").squeeze()

log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
log_reg.fit(X_train, y_train)

# Seuil par défaut (0.5) conservé : l'objectif métier priorise le recall
y_pred = log_reg.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Reste", "Parti"]))
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :\n", cm)

# ─── Coefficients ───
coefficients = pd.Series(log_reg.coef_[0], index=X_train.columns).sort_values()
print("\nCoefficients (triés) :\n", coefficients.round(3))

plt.figure(figsize=(9, 8))
colors = ["#E24B4A" if v > 0 else "#1D9E75" for v in coefficients.values]
coefficients.plot(kind="barh", color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Effet de chaque variable sur le risque de churn\n(rouge = augmente le risque, vert = le diminue)")
plt.xlabel("Coefficient (données standardisées)")
plt.tight_layout()
plt.savefig("../figures/coefficients_plot.png", dpi=150, bbox_inches="tight")
print("\nFigure sauvegardée : figures/coefficients_plot.png")
