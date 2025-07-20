#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test et visualisation yolov8 pour detection accessibilite
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import json

def load_model(model_path):
    """charge le modele yolov8"""
    if not Path(model_path).exists():
        print(f"erreur: modele non trouve: {model_path}")
        return None
    
    try:
        model = YOLO(model_path)
        print(f"modele charge: {model_path}")
        return model
    except Exception as e:
        print(f"erreur lors du chargement du modele: {e}")
        return None

def predict_image(model, image_path, conf_threshold=0.25, output_dir=None):
    """effectue la prediction sur une image"""
    
    if not Path(image_path).exists():
        print(f"erreur: image non trouvee: {image_path}")
        return None
    
    try:
        # prediction
        results = model(image_path, conf=conf_threshold)
        
        # obtenir les resultats
        result = results[0]
        
        # informations sur l'image
        img_name = Path(image_path).name
        print(f"\nimage: {img_name}")
        print(f"taille: {result.orig_shape}")
        
        # detections
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            print(f"detections trouvees: {len(boxes)}")
            
            for i, box in enumerate(boxes):
                # coordonnees
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # nom de la classe
                class_name = model.names[cls]
                
                print(f"  {i+1}. {class_name}: {conf:.3f} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        else:
            print("aucune detection trouvee")
        
        # sauvegarder la visualisation
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # image avec annotations
            annotated_img = result.plot()
            output_path = output_dir / f"prediction_{img_name}"
            cv2.imwrite(str(output_path), annotated_img)
            print(f"visualisation sauvegardee: {output_path}")
        
        return result
        
    except Exception as e:
        print(f"erreur lors de la prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_on_dataset(model, dataset_dir, output_dir, max_images=10):
    """teste le modele sur des images du dataset"""
    
    dataset_dir = Path(dataset_dir)
    val_images_dir = dataset_dir / "yolo_format" / "images" / "val"
    
    if not val_images_dir.exists():
        print(f"erreur: dossier de validation non trouve: {val_images_dir}")
        return
    
    # obtenir les images de validation
    image_files = list(val_images_dir.glob("*.jpg")) + list(val_images_dir.glob("*.png"))
    
    if not image_files:
        print(f"aucune image trouvee dans: {val_images_dir}")
        return
    
    print(f"\ntest sur {min(len(image_files), max_images)} images du dataset...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # tester sur un echantillon
    for i, img_path in enumerate(image_files[:max_images]):
        print(f"\n--- image {i+1}/{min(len(image_files), max_images)} ---")
        predict_image(model, img_path, conf_threshold=0.25, output_dir=output_dir)

def test_user_image(model, image_path, output_dir):
    """teste le modele sur une image utilisateur"""
    
    print(f"\ntest sur image utilisateur...")
    result = predict_image(model, image_path, conf_threshold=0.1, output_dir=output_dir)
    
    if result:
        print(f"test termine avec succes")
    else:
        print(f"echec du test")

def main():
    parser = argparse.ArgumentParser(description='test yolov8 pour detection accessibilite')
    parser.add_argument('--model', type=str, 
                       default='models/yolov8_accessibility_best.pt',
                       help='chemin vers le modele yolov8')
    parser.add_argument('--image', type=str,
                       help='chemin vers une image a tester')
    parser.add_argument('--dataset', type=str,
                       default='data',
                       help='dossier du dataset pour test')
    parser.add_argument('--output', type=str,
                       default='test_outputs',
                       help='dossier de sortie pour les visualisations')
    parser.add_argument('--conf', type=float,
                       default=0.25,
                       help='seuil de confiance')
    parser.add_argument('--max-images', type=int,
                       default=10,
                       help='nombre max d\'images a tester du dataset')
    
    args = parser.parse_args()
    
    # chemins absolus
    script_dir = Path(__file__).parent
    model_path = script_dir / args.model
    output_dir = script_dir / args.output
    
    print("=== test yolov8 ===")
    print(f"modele: {model_path}")
    print(f"sortie: {output_dir}")
    
    # charger le modele
    model = load_model(model_path)
    if not model:
        return
    
    # afficher les classes
    print(f"classes: {model.names}")
    
    # test selon les arguments
    if args.image:
        # test sur une image specifique
        test_user_image(model, args.image, output_dir)
    else:
        # test sur le dataset
        test_on_dataset(model, script_dir / args.dataset, output_dir, args.max_images)

if __name__ == "__main__":
    main()
