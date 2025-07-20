#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entrainement yolov8 pour detection accessibilite
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

def train_yolov8():
    """entraine un modele yolov8 sur le dataset d'accessibilite"""
    
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data" / "yolo_format"
    config_file = data_dir / "config.yaml"
    models_dir = script_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("=== entrainement yolov8 ===")
    print(f"configuration: {config_file}")
    print(f"gpu disponible: {torch.cuda.is_available()}")
    
    # verifier que le fichier de config existe
    if not config_file.exists():
        print(f"erreur: fichier de configuration non trouve: {config_file}")
        print("executez d'abord convert_to_yolov8.py")
        return
    
    # charger le modele pre-entraine yolov8
    print("\nchargement du modele yolov8n pre-entraine...")
    model = YOLO('yolov8n.pt')  # nano version pour rapidite
    
    # parametres d'entrainement
    train_params = {
        'data': str(config_file),
        'epochs': 50,
        'imgsz': 640,
        'batch': 16,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        'save': True,
        'save_period': -1,
        'cache': False,
        'device': '',  # auto-detection
        'workers': 8,
        'project': str(models_dir),
        'name': 'yolov8_accessibility',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 0,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'close_mosaic': 10,
        'resume': False,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'split': 'val',
        'save_json': False,
        'save_hybrid': False,
        'conf': None,
        'iou': 0.7,
        'max_det': 300,
        'half': False,
        'dnn': False,
        'plots': True,
        'source': None,
        'vid_stride': 1,
        'stream_buffer': False,
        'visualize': False,
        'augment': False,
        'agnostic_nms': False,
        'classes': None,
        'retina_masks': False,
        'embed': None,
        'show': False,
        'save_frames': False,
        'save_txt': False,
        'save_conf': False,
        'save_crop': False,
        'show_labels': True,
        'show_conf': True,
        'show_boxes': True,
        'line_width': None,
    }
    
    # simplifier les parametres pour eviter les erreurs
    simple_params = {
        'data': str(config_file),
        'epochs': 50,
        'imgsz': 640,
        'batch': 16,
        'lr0': 0.01,
        'device': '',  # auto-detection gpu/cpu
        'project': str(models_dir),
        'name': 'yolov8_accessibility',
        'exist_ok': True,
        'verbose': True,
        'plots': True,
        'save': True,
        'val': True
    }
    
    print(f"\nparametres d'entrainement:")
    for key, value in simple_params.items():
        print(f"  {key}: {value}")
    
    try:
        print("\ndemarrage de l'entrainement...")
        results = model.train(**simple_params)
        
        print(f"\nentrainement termine !")
        print(f"resultats: {results}")
        
        # sauvegarder le modele final
        best_model_path = models_dir / "yolov8_accessibility" / "weights" / "best.pt"
        final_model_path = models_dir / "yolov8_accessibility_best.pt"
        
        if best_model_path.exists():
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            print(f"modele sauvegarde: {final_model_path}")
        
        # afficher les metriques finales
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            print(f"\nmetriques finales:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"erreur lors de l'entrainement: {e}")
        print(f"type d'erreur: {type(e)}")
        import traceback
        traceback.print_exc()

def validate_yolov8():
    """valide le modele entraine"""
    
    script_dir = Path(__file__).parent
    models_dir = script_dir / "models"
    model_path = models_dir / "yolov8_accessibility_best.pt"
    config_file = script_dir / "data" / "yolo_format" / "config.yaml"
    
    if not model_path.exists():
        print(f"modele non trouve: {model_path}")
        return
    
    print(f"\n=== validation du modele ===")
    print(f"modele: {model_path}")
    
    try:
        model = YOLO(str(model_path))
        results = model.val(data=str(config_file))
        
        print(f"validation terminee !")
        print(f"resultats: {results}")
        
    except Exception as e:
        print(f"erreur lors de la validation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate_yolov8()
    else:
        train_yolov8()
