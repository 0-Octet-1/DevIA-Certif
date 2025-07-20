#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test yolov8 sur image utilisateur
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import cv2

def test_image_with_yolov8(image_path):
    """teste yolov8 sur une image specifique"""
    
    script_dir = Path(__file__).parent
    output_dir = script_dir / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"erreur: image non trouvee: {image_path}")
        return
    
    print(f"=== test yolov8 sur image utilisateur ===")
    print(f"image: {image_path}")
    print(f"sortie: {output_dir}")
    
    # charger le modele pre-entraine yolov8
    model = YOLO('yolov8n.pt')
    print(f"modele: yolov8n pre-entraine")
    print(f"classes disponibles: {len(model.names)} classes")
    
    # prediction avec seuil bas pour voir plus de detections
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
    else:
        print(f"\naucune detection trouvee (seuil: 0.25)")
    
    # sauvegarder la visualisation
    annotated_img = result.plot()
    output_name = f"yolov8_detection_{image_path.name}"
    output_path = output_dir / output_name
    cv2.imwrite(str(output_path), annotated_img)
    
    print(f"\nvisualisation sauvegardee: {output_path}")
    print(f"test termine avec succes !")
    
    return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python test_image_yolov8.py <chemin_image>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_image_with_yolov8(image_path)
