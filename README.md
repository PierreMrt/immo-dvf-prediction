# immo-dvf-prediction

**Prédiction de prix immobiliers pour appartements à Angers**

Basée sur données DVF (Demande de Valeurs Foncières) officielles + Machine Learning + Agents IA

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Objectifs

- **Prédire le prix** d'un appartement à Angers en fonction de ses caractéristiques
- **Analyser le marché** immobilier par quartier (maille IRIS)
- **Intégrer le DPE** (Diagnostic de Performance Énergétique) comme feature
- **Extraire automatiquement** les caractéristiques d'une annonce via agent IA
- **Comparer** le prix d'une annonce avec le marché local

## 📊 Données utilisées

| Source | Description | Lien |
|--------|-------------|------|
| **DVF géolocalisées** | Ventes immobilières officielles (DGFiP) | [data.gouv.fr](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres-geolocalisees/) |
| **DPE** | Diagnostic de Performance Énergétique | [data.ademe.fr](https://data.ademe.fr/datasets/dpe-france) |
| **IRIS** | Codes quartiers Insee | [data.gouv.fr](https://www.data.gouv.fr/datasets/historique-des-codes-iris/) |
| **BPE** | Base Permanente des Équipements | [data.gouv.fr](https://www.data.gouv.fr/datasets/base-permanente-des-equipements/) |
| **API COPRO** | Copropriétés (charges, procédures) | [data.gouv.fr](https://www.data.gouv.fr/datasets/6084d4d0b752764f35006465/) |

## 🏗️ Architecture

```
immo-dvf-prediction/
├── data/
│   ├── raw/           # Données brutes
│   └── processed/     # Données nettoyées + features
├── src/
│   ├── data/          # Extraction & preprocessing
│   ├── models/        # ML (entraînement, prédiction)
│   ├── agents/        # Scraping + parsing annonces
│   ├── utils/         # Configuration, logging
│   └── visualization/ # Graphiques, cartes
├── app/
│   └── streamlit_app.py  # Application web
└── models/            # Modèles sauvegardés
```

## ⚡ Installation

```bash
# Cloner le repo
git clone https://github.com/PierreMrt/immo-dvf-prediction.git
cd immo-dvf-prediction

# Installer dépendances
pip install -r requirements.txt

# Copier .env.example → .env et remplir les variables d'environnement
cp .env.example .env
```

## 🚀 Utilisation

### 1. Télécharger les données

```bash
make data-download
```

### 2. Nettoyer et générer les features

```bash
make data-clean
make data-features
```

### 3. Entraîner le modèle

```bash
make train
```

### 4. Lancer l'application Streamlit

```bash
make run-app
# Ou: streamlit run app/streamlit_app.py
```

Ouvrir http://localhost:8501

## 📈 Métriques du modèle (estimées)

| Modèle | MAE (€/m²) | R² |
|--------|------------|-----|
| XGBoost | ~150–200 | ~0.75–0.85 |

## 🛠️ Stack technique

- **Langage** : Python 3.11+
- **ML** : scikit-learn, XGBoost, LightGBM
- **Data** : pandas, polars, geopandas
- **Web** : Streamlit
- **Agents IA** : LangChain, OpenAI GPT
- **Visualisation** : Plotly, Matplotlib, Seaborn
- **Format** : Parquet, GeoJSON

## 📄 License

MIT License — voir [LICENSE](LICENSE) pour détails

## 👤 Auteur

[@PierreMrt](https://github.com/PierreMrt)
