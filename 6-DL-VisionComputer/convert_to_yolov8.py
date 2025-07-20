#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
conversion des donnees au format yolov8
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image
import random

def convert_to_yolov8_format():
    """convertit les annotations json vers le format yolov8"""
    
    # chemins
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    annotations_file = data_dir / "parsed_annotations.json"
    images_dir = data_dir / "images"
    
    # dossier de sortie yolov8
    yolo_dir = data_dir / "yolo_format"
    yolo_dir.mkdir(exist_ok=True)
    
    # structure yolov8
    for split in ['train', 'val']:
        (yolo_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # mapping des classes
    class_mapping = {
        'step': 0,
        'stair': 1, 
        'ramp': 2,
        'handrail': 3,
        'grab_bar': 4
    }
    
    print(f"chargement annotations: {annotations_file}")
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', [])
    valid_annotations = []
    
    # filtrer les annotations valides
    for annotation in annotations:
        if isinstance(annotation, dict) and annotation.get('boxes'):
            img_name = annotation.get('image_name')
            img_path = images_dir / img_name
            
            if img_path.exists():
                valid_annotations.append(annotation)
    
    print(f"annotations valides trouvees: {len(valid_annotations)}")
    
    # melanger et diviser train/val (80/20)
    random.shuffle(valid_annotations)
    split_idx = int(0.8 * len(valid_annotations))
    train_annotations = valid_annotations[:split_idx]
    val_annotations = valid_annotations[split_idx:]
    
    print(f"train: {len(train_annotations)}, val: {len(val_annotations)}")
    
    # traiter chaque split
    for split_name, split_annotations in [('train', train_annotations), ('val', val_annotations)]:
        print(f"\ntraitement {split_name}...")
        
        for annotation in split_annotations:
            img_name = annotation['image_name']
            img_path = images_dir / img_name
            
            # copier l'image
            dest_img = yolo_dir / 'images' / split_name / img_name
            shutil.copy2(img_path, dest_img)
            
            # creer le fichier label
            label_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = yolo_dir / 'labels' / split_name / label_name
            
            # obtenir dimensions image
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            # convertir les boites
            yolo_labels = []
            for box_data in annotation['boxes']:
                label = box_data['label']
                if label not in class_mapping:
                    print(f"warning: classe inconnue '{label}' ignoree")
                    continue
                
                class_id = class_mapping[label]
                bbox = box_data['bbox_abs']  # [x, y, w, h] en pixels
                
                # convertir en format yolo (centre x, centre y, largeur, hauteur) normalise
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height
                
                # verifier que les valeurs sont dans [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # sauvegarder le fichier label
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))
    
    # creer le fichier de configuration yaml
    config_content = f"""# configuration dataset yolov8
path: {yolo_dir.absolute()}
train: images/train
val: images/val

# classes
nc: 5
names: ['step', 'stair', 'ramp', 'handrail', 'grab_bar']
"""
    
    config_path = yolo_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"\nconversion terminee !")
    print(f"donnees yolov8: {yolo_dir}")
    print(f"configuration: {config_path}")
    print(f"classes: {class_mapping}")

if __name__ == "__main__":
    convert_to_yolov8_format()
