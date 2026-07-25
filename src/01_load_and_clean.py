"""
Étape 1-2 : chargement et nettoyage du jeu de données réel (IBM Telco Customer Churn).

Source : jeu de données public largement utilisé en Data Science
(7 043 clients, 21 variables), disponible notamment sur Kaggle.
"""
import pandas as pd

df = pd.read_csv("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("Dimensions brutes :", df.shape)

# ─── Piège de qualité de données identifié : TotalCharges est en texte ───
# 11 clients ont TotalCharges = ' ' (espace), tous avec tenure = 0
# (clients tout juste arrivés, sans facture totale encore émise)
espaces_caches = df["TotalCharges"].str.strip() == ""
print("Valeurs 'vides déguisées' détectées :", espaces_caches.sum())

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])
print("Dimensions après nettoyage :", df.shape)

taux_churn = (df["Churn"] == "Yes").mean() * 100
print(f"Taux de churn global : {taux_churn:.2f} %")

df.to_csv("../data/telco_clean.csv", index=False)
