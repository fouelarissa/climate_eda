# Dashboard Climatique - Application Streamlit

## Description

Application d'analyse et de visualisation des donnees climatiques basee sur Streamlit. Cette application permet d'explorer les donnees meteorologiques par region et saison.

## Structure du Projet

```
SN_Python/
├── app.py                      # Application principale Streamlit (FRONTEND)
├── data_processor.py           # Traitement des donnees (BACKEND)
├── visualizations.py           # Graphiques interactifs Plotly (BACKEND)
├── auto_interpretation.py      # Interpretations automatiques (BACKEND)
├── statistical_analysis.py     # Analyses statistiques avancees (BACKEND)
├── report_generator.py         # Generation de rapports PDF/Excel/HTML (BACKEND)
├── predictive_models.py        # Modeles predictifs avances ML (BACKEND)
├── api.py                      # API REST FastAPI (BACKEND - Phase 6)
├── requirements.txt            # Dependances Python
├── dataset_climat.csv          # Dataset source
└── README.md                   # Documentation
```

## Architecture

### Frontend (app.py)
- Interface utilisateur avec Streamlit
- Mise en page responsive avec colonnes
- Filtres interactifs (region, saison)
- Barre de recherche par variable
- Tableaux de donnees dynamiques
- Indicateurs metriques
- Onglets d'analyses statistiques

### Backend
- **data_processor.py** : Chargement, validation, filtrage, aggregation
- **visualizations.py** : 16 graphiques interactifs Plotly
- **auto_interpretation.py** : Interpretations contextualisees basees sur les donnees
- **statistical_analysis.py** : Tests statistiques, regressions, detection d'outliers
- **report_generator.py** : Export de rapports Excel (multi-feuilles), PDF (avec graphiques) et HTML
- **predictive_models.py** : Modeles predictifs (Random Forest, Gradient Boosting, XGBoost, Regression Lineaire)
- **api.py** : API REST FastAPI avec endpoints donnees, stats, predictions, rapports

## Installation

1. Installer les dependances :
```bash
pip install -r requirements.txt
```

2. Lancer l'application Streamlit :
```bash
streamlit run app.py
```

3. Lancer l'API REST (Phase 6) :
```bash
python api.py
```
L'API est accessible sur `http://localhost:8000` — documentation interactive sur `/docs`.

## Fonctionnalites

### Phase 1 : Structure et Donnees
- ✅ Chargement automatique du dataset
- ✅ Validation de la qualite des donnees
- ✅ Affichage des informations generales (metriques)
- ✅ Tableau de donnees interactif avec filtres
- ✅ Filtres par region et saison
- ✅ Statistiques descriptives
- ✅ Agregation par region/saison

### Phase 2 : Visualisations et Recherche
- ✅ **Barre de recherche intelligente** par variable (temperature, humidite, etc.)
- ✅ **16 graphiques interactifs** Plotly avec interpretations automatiques
- ✅ Distributions, boxplots, scatter plots, heatmaps, violin plots
- ✅ Graphiques pour variables qualitatives (barplots, pie charts)
- ✅ Croisements region x saison (empile et groupe)
- ✅ Matrice de correlation et scatter matrix
- ✅ Interpretations **contextualisees** generees automatiquement

### Phase 3 : Analyses Statistiques Avancees
- ✅ **Resumes statistiques complets** (moyenne, mediane, ecart-type, CV, skewness, kurtosis)
- ✅ **Tests d'hypotheses** :
  - Test de normalite (Shapiro-Wilk)
  - Test d'homogeneite des variances (Levene)
  - T-test de Student (2 groupes)
  - ANOVA a un facteur (plusieurs groupes)
  - Test du Chi2 d'independance
- ✅ **Regressions** :
  - Lineaire simple (avec graphique et equation)
  - Multiple (avec variables categorielles encodees)
- ✅ **Detection d'outliers** par methode IQR avec details

### Phase 5 : Modeles Predictifs Avances
- ✅ **Comparaison automatique** de modeles (Regression Lineaire, Random Forest, Gradient Boosting, XGBoost)
- ✅ **Importance des variables** avec graphique interactif
- ✅ **Performances** : R², RMSE, MAE + interpretation automatique
- ✅ **Scatter reel vs predit** avec ligne de reference
- ✅ **Prediction interactive** : entrer des valeurs et obtenir une prediction
- ✅ **Detection du surapprentissage** (overfitting) via ecart train/test

### Phase 4 : Export de Rapports
- ✅ **Rapport Excel** multi-feuilles (donnees, stats, correlations, par region, par saison)
- ✅ **Rapport PDF** avec statistiques, tableaux et graphiques (matplotlib + kaleido)
- ✅ **Rapport HTML** interactif avec tableaux et mise en forme CSS
- ✅ Telechargement direct depuis l'interface Streamlit

## Donnees

Le dataset contient 501 enregistrements avec les variables :
- `humidite` : Taux d'humidite (%)
- `precipitations` : Quantite de precipitations (mm)
- `vitesse_vent` : Vitesse du vent (km/h)
- `region` : Zone geographique (Nord, Sud, Est, Ouest, Centre)
- `saison` : Periode (saison_des_pluies, saison_sèche)
- `temperature_moyenne` : Temperature moyenne (°C)

### Phase 6 : API REST / Mise en Production
- ✅ **API FastAPI** avec documentation interactive auto-generee (`/docs`, `/redoc`)
- ✅ **Endpoints REST** :
  - `GET /health` — Etat du service et du dataset
  - `GET /data` — Donnees filtrables (region, saison, limite)
  - `GET /stats/descriptive` — Statistiques descriptives JSON
  - `GET /stats/correlation` — Matrice de correlation JSON
  - `GET /stats/by-region` et `/by-saison` — Agregations
  - `GET /tests/normality/{variable}` — Test de Shapiro-Wilk
  - `GET /tests/homogeneity/{variable}` — Test de Levene
  - `POST /models/train` — Entrainement d'un modele predictif
  - `POST /models/compare` — Comparaison automatique de modeles
  - `POST /predict` — Prediction sur une nouvelle observation
  - `GET /report/excel` — Telechargement rapport Excel
  - `GET /report/html` — Rapport HTML complet
- ✅ **Modeles Pydantic** pour validation des requetes/reponses

## Phases Futures

Toutes les phases (1 a 6) sont implementees et operationnelles.
