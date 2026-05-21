# Prédiction du Risque d'Accident Grave de Vélo en France

> Projet de Data Science appliqué à la sécurité routière — Pipeline ML complet sur +500 000 accidents réels

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![Regression Logistique](https://img.shields.io/badge/Model-Logistic_Regression-blue?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
[![SVM](https://img.shields.io/badge/Model-SVM-blueviolet?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/modules/svm.html)
[![Random Forest](https://img.shields.io/badge/Model-Random_Forest-green?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
[![Dataset](https://img.shields.io/badge/Data-data.gouv.fr-green)](https://www.data.gouv.fr/datasets/accidents-de-velo)

---

## Objectif

Développer un pipeline de Machine Learning capable de **prédire la gravité d'un accident de vélo** à partir de variables contextuelles (météo, horaire, localisation, infrastructure routière).

**L’objectif principal** était de développer un pipeline complet capable de :

- analyser plus de **80 000 accidents de vélo** enregistrés en France ;
- identifier les facteurs les plus liés à la gravité des accidents ;
- prédire le risque d’accident grave à partir de variables contextuelles ;
- comparer plusieurs modèles de Machine Learning ;
- optimiser les performances dans un contexte de données volumineuses.
- Démontrer la capacité d’un pipeline ML à traiter des données réelles à grande échelle.
  
**Objectifs métier :**
- Mieux comprendre les contextes à risque pour les cyclistes ;
- Identifier les variables influençant les accidents graves ;
- Aider à prioriser les actions de prévention ;
---

## Dataset

| Caractéristique | Détail |
|---|---|
| Source | [Data.gouv.fr — Accidents corporels](https://www.data.gouv.fr/datasets/accidents-de-velo) |
| Période | 2005 → 2023 |
| Volume | +80 000 lignes |
| Type | Données multi-annuelles, météo, temporelles, géographiques |
| Quelques variables |luminosité (`lum`), météo (`atm`), sexe (`sexe`), heure (`hrmn`) ... |

**Variable cible (binaire) :**
- `0` → Accident léger
- `1` → Accident grave
---

## Pipeline Data Science
**Collecte → Nettoyage → EDA → Feature Engineering → Modélisation → Évaluation → Optimisation**

**1. Nettoyage des données**
- gestion des valeurs manquantes et incohérences ;
- harmonisation des variables ;
- transformation des types ;
- création de la variable cible binaire.

**2. Analyse Exploratoire des Données (EDA)**
L’EDA a permis d’identifier plusieurs tendances importantes.
- Les accidents augmentent fortement sur certaines plages horaires.
- Les accidents graves sont plus fréquents hors agglomération.
- Les conditions météorologiques influencent la gravité.
- Certains départements présentent des volumes très élevés.
- Les accidents graves sont minoritaires → problème de déséquilibre de classes.

**3. Feature engineering (saison, heure, département)** et - Traitement du déséquilibre de classes
  
**4. Entraînement et comparaison de 5 modèles ML**
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
## 🎥 Présentation vidéo du projet

Une présentation de 06 minutes résumant le contexte, la démarche et les principaux résultats.

👉 [Voir la vidéo de présentation](https://www.canva.com/design/DAHKTgqJuaQ/5HF3ZWtyt6Y1H_Lo0KveOQ/view?utm_content=DAHKTgqJuaQ&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hd1f0429750) 👈

---

## 📈 Visuels clés du projet
### 1. Évolution annuelle des accidents
➡ Montre l’analyse temporelle du phénomène.

**Capture recommandée :**
- courbe annuelle des accidents 2005–2023.


### 2. Volume vs taux de gravité selon l’heure
➡ Très fort impact visuel et métier.

Pourquoi c’est important :
- démontre la double lecture volume + dangerosité ;
- montre une vraie réflexion analytique.

### 3. Analyse météo / luminosité / surface
➡ Très bon visuel métier.

À capturer :
- les graphiques comparant les conditions de circulation

### 4. Top départements accidentogènes
➡ Excellent pour montrer l’analyse géographique.

### 5. Heatmap de corrélation
➡ Indispensable pour montrer la compréhension statistique.

C’est un visuel très attendu dans un projet Data Science professionnel.

### 6. Comparaison des modèles ML
➡ Capture essentielle.

À montrer :
- Accuracy
- Recall
- F1-score
- ROC AUC

Ce graphique prouve la démarche scientifique de comparaison des modèles.

### 7. Courbes ROC
➡ Très important pour un portfolio ML.

Pourquoi :
- montre la maîtrise de l’évaluation probabiliste ;
- visuel très professionnel.

### 8. Matrices de confusion
➡ Excellente preuve de compréhension métier.

À privilégier :
- matrices avec effectifs + pourcentages.

### 9. Optimisation du seuil de décision
➡ Très différenciant pour un recruteur.

Cette partie montre que le projet ne s’arrête pas à l’entraînement du modèle.

### 10. Feature Importance
➡ Probablement l’un des meilleurs visuels du notebook.

À capturer :
- importance des variables Random Forest ;
- coefficients Logistic Regression.

Cela démontre :
- interprétabilité ;
- compréhension métier ;
- capacité à expliquer un modèle.


### Comparaison des modèles
![Comparaison des modèles](img/Comparaison_des_models.png)

### Feature Importance — Random Forest
![Feature Importance](img/Comparaison_des_modelsROC.png)

### Matrice de confusion — XGBoost
![Matrice de confusion](img/matrices_confusion_XGB.png)

### Distribution temporelle des accidents
![EDA temporelle](img/evolution_annuelle.png)

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

```bash
accidents-velo-prediction
├── notebook.ipynb ← Pipeline complet
├── README.md
├── data/
│ └── readme.md ← Source et description du dataset
├── images/
│ └── .png ← Captures des résultats
```


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


