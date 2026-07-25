"""
Étape 2 : analyse exploratoire des données (dataset réel Telco).
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/telco_clean.csv")
df["churn_binaire"] = (df["Churn"] == "Yes").astype(int)


def taux_churn_par_categorie(colonne):
    return (
        df.groupby(colonne)["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
    ).round(1).sort_values(ascending=False)


print("Par type de contrat :\n", taux_churn_par_categorie("Contract"))
print("\nPar service internet :\n", taux_churn_par_categorie("InternetService"))
print("\nPar mode de paiement :\n", taux_churn_par_categorie("PaymentMethod"))
print("\nSelon senior citizen :\n", taux_churn_par_categorie("SeniorCitizen"))

print(
    "\nCorrélation tenure / TotalCharges :",
    df["tenure"].corr(df["TotalCharges"]).round(3),
)

colonnes_numeriques = ["tenure", "MonthlyCharges", "TotalCharges"]
correlations = df[colonnes_numeriques + ["churn_binaire"]].corr()["churn_binaire"].drop("churn_binaire")
print("\nCorrélations avec le churn :\n", correlations.sort_values(ascending=False).round(3))

# Voir figures/eda_overview.png pour la version visuelle de cette exploration
