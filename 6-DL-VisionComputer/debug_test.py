#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script de debug pour tester le modele yolo
"""

import sys
import importlib.util
from pathlib import Path
import torch

# import du modele
script_dir = Path(__file__).parent
spec = importlib.util.spec_from_file_location("modele_yolo", script_dir / "3-modele_yolo.py")
yolo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(yolo_module)
accessibility_yolo = yolo_module.accessibility_yolo

# import du post-processing
from yolo_postprocessing import decode_yolo_predictions, analyze_accessibility_from_detections

def debug_prediction(image_path):
    """test de debug pour une image"""
    print(f"=== debug prediction ===")
    print(f"image: {image_path}")
    
    # charger le modele
    model = accessibility_yolo(num_classes=4)
    model_path = script_dir / "models" / "yolo_accessibility.pth"
    
    if not model_path.exists():
        print(f"erreur: modele non trouve: {model_path}")
        return
    
    print(f"chargement modele: {model_path}")
    model.load_model(str(model_path))
    
    # prediction
    print("debut prediction...")
    result = model.predict(image_path)
    print(f"prediction terminee")
    
    if 'predictions_tensor' in result:
        predictions = result['predictions_tensor']
        print(f"predictions shape: {predictions.shape}")
        print(f"predictions min/max: {predictions.min():.4f} / {predictions.max():.4f}")
        
        # test avec differents seuils
        for threshold in [0.01, 0.05, 0.1, 0.2]:
            print(f"\n--- seuil {threshold} ---")
            detections = decode_yolo_predictions(predictions, confidence_threshold=threshold)
            print(f"detections: {len(detections)}")
            
            for det in detections:
                print(f"  {det['label']}: {det['confidence']:.3f}")
    
    else:
        print(f"erreur: {result.get('message', 'format invalide')}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python debug_test.py <image_path>")
        sys.exit(1)
    
    debug_prediction(sys.argv[1])
