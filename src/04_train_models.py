"""
Étape 4-5 : entraînement des modèles et comparaison par validation croisée.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

X_train = pd.read_csv("../data/X_train.csv")
X_test = pd.read_csv("../data/X_test.csv")
y_train = pd.read_csv("../data/y_train.csv").squeeze()
y_test = pd.read_csv("../data/y_test.csv").squeeze()

X_full = pd.concat([X_train, X_test], ignore_index=True)
y_full = pd.concat([y_train, y_test], ignore_index=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─── Modèle 1 : régression logistique ───
log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

print("=== Régression logistique ===")
print(classification_report(y_test, y_pred_log, target_names=["Reste", "Parti"]))
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred_log))

scores_log = cross_val_score(log_reg, X_full, y_full, cv=cv, scoring="recall")
print(f"Recall moyen (5-fold) : {scores_log.mean():.3f} (+/- {scores_log.std():.3f})")

# ─── Modèle 2 : Random Forest ───
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n=== Random Forest ===")
print(classification_report(y_test, y_pred_rf, target_names=["Reste", "Parti"]))
scores_rf = cross_val_score(rf, X_full, y_full, cv=cv, scoring="recall")
print(f"Recall moyen (5-fold) : {scores_rf.mean():.3f} (+/- {scores_rf.std():.3f})")

# ─── Modèle 3 : Gradient Boosting ───
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

print("\n=== Gradient Boosting ===")
print(classification_report(y_test, y_pred_gb, target_names=["Reste", "Parti"]))
scores_gb = cross_val_score(gb, X_full, y_full, cv=cv, scoring="recall")
print(f"Recall moyen (5-fold) : {scores_gb.mean():.3f} (+/- {scores_gb.std():.3f})")

print("\n--> Modèle retenu : régression logistique (meilleur recall, le plus stable)")
