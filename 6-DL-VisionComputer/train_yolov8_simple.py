#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entrainement yolov8 simplifie pour eviter les problemes de compatibilite
"""

from pathlib import Path
from ultralytics import YOLO
import torch

def train_simple():
    """entrainement yolov8 avec parametres minimaux"""
    
    script_dir = Path(__file__).parent
    config_file = script_dir / "data" / "yolo_format" / "config.yaml"
    
    if not config_file.exists():
        print(f"erreur: config non trouvee: {config_file}")
        return
    
    print("=== entrainement yolov8 simplifie ===")
    print(f"config: {config_file}")
    print(f"gpu: {torch.cuda.is_available()}")
    
    try:
        # modele yolov8 nano
        model = YOLO('yolov8n.pt')
        
        # entrainement avec parametres minimaux
        results = model.train(
            data=str(config_file),
            epochs=10,  # reduit pour rapidite
            imgsz=640,
            batch=8,    # reduit pour eviter les problemes memoire
            device='',  # auto-detection
            project='models',
            name='yolov8_accessibility_simple',
            exist_ok=True,
            verbose=False,  # moins de logs
            plots=False,    # pas de plots pour eviter pandas
            save=True,
            val=True
        )
        
        print(f"entrainement termine !")
        
        # sauvegarder le modele
        model_path = script_dir / "models" / "yolov8_accessibility_simple" / "weights" / "best.pt"
        if model_path.exists():
            final_path = script_dir / "models" / "yolov8_accessibility_final.pt"
            import shutil
            shutil.copy2(model_path, final_path)
            print(f"modele sauvegarde: {final_path}")
        
    except Exception as e:
        print(f"erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_simple()
