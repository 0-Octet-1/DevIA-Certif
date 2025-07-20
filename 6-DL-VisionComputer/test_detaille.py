#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test detaille du modele yolov8 entraine avec affichage complet
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

def test_detaille(image_path):
    """test detaille avec affichage complet des resultats"""
    
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
        return
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"erreur: image non trouvee: {image_path}")
        return
    
    print("=" * 60)
    print("TEST DETAILLE YOLOV8 ENTRAINE")
    print("=" * 60)
    print(f"modele: {model_path.name}")
    print(f"image: {image_path.name}")
    print(f"taille image: {image_path.stat().st_size} bytes")
    
    # charger le modele
    model = YOLO(str(model_path))
    print(f"\nclasses du modele entraine:")
    for i, class_name in model.names.items():
        print(f"  {i}: {class_name}")
    
    # charger l'image pour info
    img = cv2.imread(str(image_path))
    if img is not None:
        h, w, c = img.shape
        print(f"\ndimensions image: {w}x{h} pixels")
    
    print("\n" + "=" * 60)
    print("PREDICTIONS AVEC DIFFERENTS SEUILS")
    print("=" * 60)
    
    # test avec plusieurs seuils
    seuils = [0.1, 0.2, 0.3, 0.5]
    
    for seuil in seuils:
        print(f"\n--- SEUIL {seuil} ---")
        results = model(str(image_path), conf=seuil, iou=0.45, verbose=False)
        result = results[0]
        
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            print(f"detections trouvees: {len(boxes)}")
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = model.names[cls]
                
                # calculer taille de la boite
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                print(f"  {i+1}. {class_name}")
                print(f"     confiance: {conf:.3f}")
                print(f"     position: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
                print(f"     taille: {width:.0f}x{height:.0f} pixels ({area:.0f} px²)")
                
                # verification classe accessibilite
                if class_name in ['step', 'stair', 'ramp', 'handrail', 'grab_bar']:
                    print(f"     >>> CLASSE ACCESSIBILITE DETECTEE ! <<<")
        else:
            print("aucune detection")
    
    # sauvegarder visualisation avec seuil optimal
    print(f"\n" + "=" * 60)
    print("SAUVEGARDE VISUALISATION")
    print("=" * 60)
    
    # utiliser seuil 0.1 pour voir toutes les detections
    results_final = model(str(image_path), conf=0.1, iou=0.45, verbose=False)
    result_final = results_final[0]
    
    # creer visualisation
    annotated_img = result_final.plot(
        conf=True,          # afficher confiance
        labels=True,        # afficher labels
        boxes=True,         # afficher boites
        line_width=2,       # epaisseur lignes
        font_size=12        # taille police
    )
    
    # sauvegarder
    output_name = f"test_detaille_{image_path.name}"
    output_path = output_dir / output_name
    cv2.imwrite(str(output_path), annotated_img)
    
    print(f"visualisation sauvegardee: {output_path}")
    print(f"taille fichier: {output_path.stat().st_size} bytes")
    
    # afficher resume final
    print(f"\n" + "=" * 60)
    print("RESUME FINAL")
    print("=" * 60)
    
    boxes_final = result_final.boxes
    if boxes_final is not None and len(boxes_final) > 0:
        classes_detectees = set()
        for box in boxes_final:
            cls = int(box.cls[0].cpu().numpy())
            class_name = model.names[cls]
            classes_detectees.add(class_name)
        
        print(f"nombre total de detections: {len(boxes_final)}")
        print(f"classes detectees: {', '.join(sorted(classes_detectees))}")
        
        # compter classes accessibilite
        classes_accessibilite = [c for c in classes_detectees if c in ['step', 'stair', 'ramp', 'handrail', 'grab_bar']]
        if classes_accessibilite:
            print(f"classes accessibilite: {', '.join(classes_accessibilite)}")
            print(">>> SUCCES: ELEMENTS ACCESSIBILITE DETECTES ! <<<")
        else:
            print("aucune classe accessibilite detectee")
    else:
        print("aucune detection trouvee")
    
    print("=" * 60)
    return result_final

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python test_detaille.py <chemin_image>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_detaille(image_path)
