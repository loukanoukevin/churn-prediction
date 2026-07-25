# Prédiction du churn client

Projet de machine learning supervisé de bout en bout : prédire quels clients
risquent de résilier leur abonnement, à partir de leurs caractéristiques
contractuelles et comportementales.

**Jeu de données** : [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(7 043 clients, 21 variables) — un jeu de données public de référence pour ce
type de problème.

## Résultats clés

| Modèle | Recall moyen (validation croisée 5-fold) | Stabilité |
|---|---|---|
| **Régression logistique** | **0.805** | ± 0.022 |
| Random Forest | 0.477 | ± 0.018 |
| Gradient Boosting | 0.523 | ± 0.023 |

Le modèle final (régression logistique) détecte **79 % des clients qui
résilient réellement**, avec une precision de 49 %.

**Facteurs de risque principaux** : service internet fibre optique, paiement
par chèque électronique, contrat mensuel (sans engagement).
**Facteurs protecteurs** : contrat de deux ans, ancienneté, support technique
et sécurité en ligne souscrits.

Voir le [rapport complet](report/Rapport_Projet_Churn.pdf) pour le détail de
la démarche (cadrage, exploration, gestion d'un vrai défaut de qualité de
données, préparation, comparaison de modèles, interprétation, recommandation
business).

## Structure du repo

```
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── src/
│   ├── 01_load_and_clean.py       # chargement + traitement du défaut TotalCharges
│   ├── 02_eda.py                  # analyse exploratoire
│   ├── 03_prepare_data.py         # feature engineering, encodage, split
│   ├── 04_train_models.py         # entraînement et comparaison des 3 modèles
│   └── 05_interpret_model.py      # interprétation du modèle final + figure
├── figures/
└── report/
    ├── Rapport_Projet_Churn.pdf
    └── Rapport_Projet_Churn.tex
```

## Reproduire les résultats

```bash
pip install -r requirements.txt
cd src
python 01_load_and_clean.py
python 02_eda.py
python 03_prepare_data.py
python 04_train_models.py
python 05_interpret_model.py
```

## Démarche suivie

1. **Cadrage** — définir l'objectif (détecter un maximum de départs réels) et
   la métrique prioritaire (recall plutôt qu'accuracy, sur un problème
   déséquilibré à 26,5 % de churn).
2. **Chargement et qualité des données** — détection et traitement d'un vrai
   défaut : `TotalCharges` stockée en texte avec 11 valeurs "vides déguisées"
   (espaces), invisibles à un simple `isnull()`.
3. **Analyse exploratoire** — identification des signaux forts (contrat,
   mode de paiement, service internet) et des redondances entre variables.
4. **Préparation des données** — feature engineering, encodage, standardisation.
5. **Modélisation** — comparaison de 3 modèles par validation croisée.
6. **Évaluation** — analyse du compromis precision/recall ; choix assumé de
   conserver le seuil par défaut plutôt que le seuil "optimal" au sens du F1,
   pour rester cohérent avec la priorité métier définie au cadrage.
7. **Interprétation & recommandation** — traduction des coefficients en
   profil de risque et en actions business concrètes.

## Stack technique

Python · Pandas · Scikit-learn · Matplotlib
