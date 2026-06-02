# immo-dvf-prediction - Makefile

.PHONY: help install dev lint format data-download data-clean data-join data-features data-full train evaluate run-app clean

# ============================================
# Aide
# ============================================

help: ## Afficher cette aide
	@echo "immo-dvf-prediction - Commandes disponibles:"
	@echo ""
	@echo "Installation:"
	@echo "  install    - Installer dépendances production"
	@echo "  dev        - Installer dépendances dev + linting"
	@echo ""
	@echo "Qualité code:"
	@echo "  lint       - Linting (ruff + black + mypy)"
	@echo "  format     - Formater le code (black + ruff)"
	@echo ""
	@echo "Données:"
	@echo "  data-download  - Télécharger données DVF + DPE + OSM"
	@echo "  data-clean     - Nettoyer données DVF"
	@echo "  data-join      - Jointures DVF + DPE + OSM + IRIS"
	@echo "  data-features  - Générer features engineering"
	@echo "  data-full      - Pipeline complet données (download → features)"
	@echo ""
	@echo "ML:"
	@echo "  train        - Entraîner les modèles"
	@echo "  evaluate     - Évaluer les modèles"
	@echo ""
	@echo "Application:"
	@echo "  run-app      - Lancer application Streamlit"
	@echo ""
	@echo "Nettoyage:"
	@echo "  clean        - Supprimer fichiers temporaires"

# ============================================
# Installation
# ============================================

install: ## Installer dépendances production
	pip install -r requirements.txt

dev: ## Installer dépendances dev + linting
	pip install -r requirements-dev.txt
	pre-commit install

# ============================================
# Qualité code
# ============================================

lint: ## Linting (ruff + black + mypy)
	ruff check src/
	black --check src/
	mypy src/

format: ## Formater le code
	black src/
	ruff check --fix src/

# ============================================
# Données
# ============================================

data-download: ## Télécharger données DVF + DPE + OSM
	python -m src.data.download

data-clean: ## Nettoyer données DVF
	python -m src.data.clean

data-join: ## Jointures DVF + DPE + OSM + IRIS
	python -m src.data.join

data-features: ## Générer features engineering
	python -m src.data.features

data-full: ## Pipeline complet données
	$(MAKE) data-download
	$(MAKE) data-clean
	$(MAKE) data-join
	$(MAKE) data-features

# ============================================
# ML
# ============================================

train: ## Entraîner les modèles
	python -m src.models.train

evaluate: ## Évaluer les modèles
	python -m src.models.evaluate

# ============================================
# Application
# ============================================

run-app: ## Lancer application Streamlit
	streamlit run app/streamlit_app.py

# ============================================
# Nettoyage
# ============================================

clean: ## Supprimer fichiers temporaires
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf logs/*.log
