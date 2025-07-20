#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test yolov8 pre-entraine sur nos donnees
"""

from pathlib import Path
from ultralytics import YOLO
import cv2

def test_pretrained_yolo():
    """teste yolov8 pre-entraine sur nos images"""
    
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    output_dir = script_dir / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    
    print("=== test yolov8 pre-entraine ===")
    
    # charger le modele pre-entraine
    model = YOLO('yolov8n.pt')
    print(f"classes yolov8 standard: {model.names}")
    
    # tester sur quelques images de notre dataset
    val_images = list((data_dir / "yolo_format" / "images" / "val").glob("*.jpg"))[:5]
    
    for i, img_path in enumerate(val_images):
        print(f"\n--- image {i+1}: {img_path.name} ---")
        
        # prediction
        results = model(str(img_path), conf=0.25)
        result = results[0]
        
        # afficher les detections
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            print(f"detections: {len(boxes)}")
            for j, box in enumerate(boxes):
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls]
                print(f"  {j+1}. {class_name}: {conf:.3f}")
        else:
            print("aucune detection")
        
        # sauvegarder la visualisation
        annotated_img = result.plot()
        output_path = output_dir / f"yolov8_pretrained_{img_path.name}"
        cv2.imwrite(str(output_path), annotated_img)
        print(f"sauvegarde: {output_path}")
    
    print(f"\ntest termine ! visualisations dans: {output_dir}")

if __name__ == "__main__":
    test_pretrained_yolo()
