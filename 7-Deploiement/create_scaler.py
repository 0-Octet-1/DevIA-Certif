#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRÉATION DU SCALER POUR IMAGES - ACCESSIBILITÉ PMR

Script pour créer et sauvegarder un scaler pour la normalisation des images
dans le cadre du déploiement du modèle CNN.
"""

import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from PIL import Image
import logging
import sys

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("create-scaler")

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(BASE_DIR, "..", "5-TravailDL", "computer_vision")
SCALER_PATH = os.path.join(TARGET_DIR, "scaler_images.pkl")

def generate_synthetic_images(n_samples=100, img_size=64):
    """
    Génère des images synthétiques pour créer le scaler
    
    Args:
        n_samples: Nombre d'échantillons à générer
        img_size: Taille des images (carré)
        
    Returns:
        np.array: Images synthétiques aplaties
    """
    logger.info(f"Génération de {n_samples} images synthétiques de taille {img_size}x{img_size}")
    
    # Initialisation du tableau pour stocker les images
    images = []
    
    # Génération des images synthétiques
    for i in range(n_samples):
        # Création d'une image aléatoire
        img_array = np.random.rand(img_size, img_size, 3)
        
        # Normalisation entre 0 et 1
        img_array = img_array.astype('float32') / 255.0
        
        # Aplatissement
        img_flat = img_array.flatten()
        
        # Ajout à la liste
        images.append(img_flat)
    
    return np.array(images)

def create_and_save_scaler():
    """
    Crée et sauvegarde un scaler pour la normalisation des images
    """
    try:
        # Génération des images synthétiques
        synthetic_images = generate_synthetic_images(n_samples=200, img_size=64)
        
        # Création du scaler
        logger.info("Création du scaler...")
        scaler = StandardScaler()
        scaler.fit(synthetic_images)
        
        # Création du répertoire cible si nécessaire
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        
        # Sauvegarde du scaler
        logger.info(f"Sauvegarde du scaler dans {SCALER_PATH}")
        joblib.dump(scaler, SCALER_PATH)
        
        logger.info("✅ Scaler créé et sauvegardé avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de la création du scaler: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Démarrage de la création du scaler pour images")
    success = create_and_save_scaler()
    if success:
        logger.info("✅ Processus terminé avec succès")
    else:
        logger.error("❌ Échec du processus")
        sys.exit(1)
