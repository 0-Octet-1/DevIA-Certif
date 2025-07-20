#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test du modele yolov8 entraine sur nos 5 classes d'accessibilite
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import cv2

def test_modele_entraine(image_path):
    """teste le modele yolov8 entraine sur nos classes"""
    
    script_dir = Path(__file__).parent
    output_dir = script_dir / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    
    # chercher le modele entraine
    model_paths = [
        script_dir / "models" / "yolov8_accessibility_trained.pt",
        script_dir / "models" / "yolov8_accessibility_full" / "weights" / "best.pt",
        script_dir / "models" / "yolov8_accessibility_full" / "weights" / "last.pt"
    ]
    
    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = path
            break
    
    if not model_path:
        print("erreur: aucun modele entraine trouve")
        print("chemins cherches:")
        for path in model_paths:
            print(f"  - {path}")
        return
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"erreur: image non trouvee: {image_path}")
        return
    
    print(f"=== test modele yolov8 entraine ===")
    print(f"modele: {model_path}")
    print(f"image: {image_path}")
    print(f"sortie: {output_dir}")
    
    # charger le modele entraine
    model = YOLO(str(model_path))
    print(f"classes du modele: {model.names}")
    print(f"nombre de classes: {len(model.names)}")
    
    # prediction avec seuil adapte
    print(f"\nprediction en cours...")
    results = model(str(image_path), conf=0.25, iou=0.45)
    result = results[0]
    
    # informations sur l'image
    print(f"taille image: {result.orig_shape}")
    
    # analyser les detections
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        print(f"\ndetections trouvees: {len(boxes)}")
        
        # trier par confiance decroissante
        confidences = boxes.conf.cpu().numpy()
        sorted_indices = confidences.argsort()[::-1]
        
        for i, idx in enumerate(sorted_indices):
            box = boxes[idx]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            class_name = model.names[cls]
            
            print(f"  {i+1}. {class_name}: {conf:.3f} [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
            
            # verification si c'est une classe d'accessibilite
            if class_name in ['step', 'stair', 'ramp', 'handrail', 'grab_bar']:
                print(f"    -> classe d'accessibilite detectee !")
    else:
        print(f"\naucune detection trouvee (seuil: 0.25)")
        print("essai avec seuil plus bas...")
        
        # essai avec seuil tres bas
        results_low = model(str(image_path), conf=0.1, iou=0.45)
        result_low = results_low[0]
        boxes_low = result_low.boxes
        
        if boxes_low is not None and len(boxes_low) > 0:
            print(f"detections avec seuil 0.1: {len(boxes_low)}")
            for i, box in enumerate(boxes_low):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls]
                print(f"  {i+1}. {class_name}: {conf:.3f}")
        else:
            print("aucune detection meme avec seuil 0.1")
    
    # sauvegarder la visualisation
    annotated_img = result.plot()
    output_name = f"modele_entraine_{image_path.name}"
    output_path = output_dir / output_name
    cv2.imwrite(str(output_path), annotated_img)
    
    print(f"\nvisualisation sauvegardee: {output_path}")
    print(f"test termine !")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python test_modele_entraine.py <chemin_image>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_modele_entraine(image_path)
