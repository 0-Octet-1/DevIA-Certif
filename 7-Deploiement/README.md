# AccesLibre - Déploiement FastAPI
##  Grégory LE TERTE

---

## Description

Application web FastAPI pour le déploiement du modèle de prédiction d'accessibilité PMR basé sur l'analyse d'images d'entrées d'établissements. Cette application permet d'uploader une photo d'entrée d'établissement et d'obtenir une prédiction sur son accessibilité aux personnes à mobilité réduite.

## Installation

```bash
# Installation des dépendances
pip install -r requirements.txt
```

## Lancement de l'application

```bash
# Depuis le dossier 6-Deploiement
python app.py
```

L'application sera accessible à l'adresse : http://localhost:8000

##  Structure du projet

```
6-Deploiement/
├── app.py                 # Application FastAPI principale
├── requirements.txt       # Dépendances Python
├── static/                # Fichiers statiques
│   ├── css/               # Styles CSS
│   │   └── styles.css     # Feuille de style principale
│   ├── js/                # Scripts JavaScript
│   │   └── main.js        # Script principal
│   └── images/            # Images et ressources
│       └── logo.png       # Logo de l'application
└── templates/             # Templates Jinja2
    └── index.html         # Page d'accueil
```

## 🔄 Fonctionnalités

- Upload d'images d'entrées d'établissements
- Prétraitement des images (redimensionnement, normalisation)
- Prédiction d'accessibilité PMR via modèle CNN
- Affichage des résultats avec niveau de confiance
- Interface responsive et moderne

## 🔗 Intégration avec le projet Deep Learning

Cette application utilise le modèle CNN entraîné dans le bloc 5 (Deep Learning) pour prédire l'accessibilité PMR à partir d'images. Elle charge automatiquement le modèle et le scaler depuis le dossier du projet Deep Learning.

## 📊 API REST

L'application expose également une API REST pour l'intégration avec d'autres services :

- `GET /` : Interface web principale
- `POST /analyze` : Endpoint pour l'analyse d'images
- `GET /health` : Vérification de l'état de l'API

## 🎓 Conformité avec la certification

Ce déploiement démontre la capacité à :
- Industrialiser un modèle de Deep Learning
- Créer une API REST pour exposer le modèle
- Développer une interface utilisateur intuitive
- Assurer la robustesse et la fiabilité du système

---

*Développé par Grégory LE TERTE *
