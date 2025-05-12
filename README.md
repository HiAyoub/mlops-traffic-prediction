#  MLOps Traffic Prediction Pipeline

Un projet MLOps complet pour prédire les débits de trafic à partir de données FCD (Floating Car Data) en utilisant Python, FastAPI, et Docker.

##  Objectif
Ce projet vise à :
- Traiter des données FCD issues de véhicules connectés
- Construire un modèle de prédiction de trafic (régression, clustering, etc.)
- Déployer une API REST pour faire des prédictions à la volée
- Dockeriser toute la chaîne pour un déploiement portable

##  Structure du projet
mlops-traffic/
├── data/ # Données brutes ou transformées (non suivies par Git)
├── models/ # Modèles entraînés (non suivis par Git)
├── notebooks/ # Analyses exploratoires
├── src/ # Code source
│ ├── preprocessing/ # Nettoyage & transformation
│ ├── training/ # Scripts d'entraînement ML
│ └── api/ # API FastAPI
├── tests/ # Tests unitaires
├── docker/ # Fichiers Docker/Docker-compose
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md


## 🚀 Démarrage rapide

```bash
# Cloner le projet
git clone https://github.com/HiAyoub/mlops-traffic-prediction.git
cd mlops-traffic-prediction

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'API localement (sans Docker)
uvicorn src.api.main:app --reload


