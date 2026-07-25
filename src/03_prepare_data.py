"""
Étape 3 : préparation des données (dataset réel Telco).
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/telco_clean.csv")

df["churn_binaire"] = (df["Churn"] == "Yes").astype(int)
df = df.drop(columns=["customerID", "Churn"])

# Variable dérivée : neutralise en partie la redondance tenure / TotalCharges
df["charge_moyenne_reelle"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

# Encodage des variables binaires
colonnes_binaires = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for col in colonnes_binaires:
    df[col] = df[col].map({"Yes": 1, "No": 0})
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

# Encodage des variables catégorielles à plusieurs modalités
colonnes_categorielles = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
df_encoded = pd.get_dummies(df, columns=colonnes_categorielles, drop_first=True)

X = df_encoded.drop(columns=["churn_binaire"])
y = df_encoded["churn_binaire"]

colonnes_numeriques = ["tenure", "MonthlyCharges", "TotalCharges", "charge_moyenne_reelle"]
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[colonnes_numeriques] = scaler.fit_transform(X[colonnes_numeriques])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Taille train : {X_train.shape[0]} | Taille test : {X_test.shape[0]}")
print(f"Taux de churn train : {y_train.mean()*100:.1f}% | test : {y_test.mean()*100:.1f}%")

X_train.to_csv("../data/X_train.csv", index=False)
X_test.to_csv("../data/X_test.csv", index=False)
y_train.to_csv("../data/y_train.csv", index=False)
y_test.to_csv("../data/y_test.csv", index=False)
