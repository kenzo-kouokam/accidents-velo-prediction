# Prédiction du Risque d'Accident Grave de Vélo en France

> Projet de Data Science appliqué à la sécurité routière — Pipeline ML complet sur +500 000 accidents réels

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Data-data.gouv.fr-green)](https://www.data.gouv.fr/datasets/accidents-de-velo)

---

## Objectif

Développer un pipeline de Machine Learning capable de **prédire la gravité d'un accident de vélo** à partir de variables contextuelles (météo, horaire, localisation, infrastructure routière).

**Objectifs métier :**
- Identifier les facteurs de risque les plus influents
- Aider à prioriser les actions de prévention routière
- Démontrer la capacité d'un pipeline ML à traiter des données réelles à grande échelle

---

## Dataset

| Caractéristique | Détail |
|---|---|
| Source | [Data.gouv.fr — Accidents corporels](https://www.data.gouv.fr/datasets/accidents-de-velo) |
| Période | 2005 → 2023 |
| Volume | +500 000 lignes |
| Type | Données multi-annuelles, météo, temporelles, géographiques |

**Variable cible (binaire) :**
- `0` → Accident léger
- `1` → Accident grave

---

## Pipeline Data Science
Collecte → Nettoyage → EDA → Feature Engineering → Modélisation → Évaluation → Optimisation


**Étapes réalisées :**
- Gestion des valeurs manquantes et incohérences
- Création de la variable cible binaire
- Analyse exploratoire complète (EDA)
- Feature engineering (saison, heure, département)
- Traitement du déséquilibre de classes
- Entraînement et comparaison de 5 modèles ML
- Optimisation mémoire et seuil de décision

---

## 🤖 Modèles comparés

| Modèle | Notes |
|---|---|
| Logistic Regression | Baseline interprétable |
| Random Forest | Bonne robustesse, top feature importance |
| **XGBoost** | **Meilleur recall — modèle retenu** |
| SVM / LinearSVC | Comparaison scalabilité |

---

## 📈 Résultats clés

### Comparaison des modèles
![Comparaison des modèles](img/model-results.png)

### Feature Importance — Random Forest
![Feature Importance](img/feature-importance.png)

### Matrice de confusion — XGBoost
![Matrice de confusion](img/confusion-matrix.png)

### Distribution temporelle des accidents
![EDA temporelle](img/eda-overview.png)

---

## ⚙️ Optimisations techniques

- **Saturation RAM** : passage de SVM classique → LinearSVC pour réduire les coûts mémoire
- **Traitement par batch** : optimisation des calculs sur gros volumes
- **Ajustement du seuil de décision** : optimisation métier du Recall et F1-score

---

## 🛠️ Stack technique

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn` · `Jupyter Notebook`

---

## 📂 Structure du repo
accidents-velo-prediction/
├── notebook.ipynb ← Pipeline complet
├── README.md
├── data/
│ └── README.md ← Source et description du dataset
└── img/
└── *.png ← Captures des résultats


---

## 💡 Compétences démontrées

- **Data Analysis** : EDA avancée, visualisation, analyse métier
- **Machine Learning** : classification supervisée, tuning, optimisation de seuils
- **Data Engineering léger** : optimisation mémoire, gestion de volumétrie
- **Communication Data** : storytelling analytique, restitution orientée métier

---

## 👨‍💻 Auteur

**Enzo Kouokam** — Data Analyst / Data Scientist Junior  
[LinkedIn](https://www.linkedin.com/in/enzo-kamhoua/) · [GitHub](https://github.com/kenzo-kouokam)


