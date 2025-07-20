#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entrainement yolov8 complet sur nos 5 classes d'accessibilite
"""

from pathlib import Path
from ultralytics import YOLO
import torch

def train_full():
    """entrainement yolov8 complet avec tous les parametres optimaux"""
    
    script_dir = Path(__file__).parent
    config_file = script_dir / "data" / "yolo_format" / "config.yaml"
    
    if not config_file.exists():
        print(f"erreur: config non trouvee: {config_file}")
        return
    
    print("=== entrainement yolov8 complet ===")
    print(f"config: {config_file}")
    print(f"gpu disponible: {torch.cuda.is_available()}")
    print(f"classes cibles: step, stair, ramp, handrail, grab_bar")
    
    try:
        # modele yolov8 nano pour rapidite
        model = YOLO('yolov8n.pt')
        print(f"modele base: yolov8n (transfer learning)")
        
        # parametres d'entrainement optimaux
        results = model.train(
            data=str(config_file),
            epochs=80,          # entrainement complet
            imgsz=640,
            batch=16,           # batch size optimal
            lr0=0.01,           # learning rate initial
            lrf=0.01,           # learning rate final
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,            # poids loss boites
            cls=0.5,            # poids loss classification
            dfl=1.5,            # poids loss distribution focal
            device='',          # auto-detection gpu/cpu
            project='models',
            name='yolov8_accessibility_full',
            exist_ok=True,
            verbose=True,
            plots=True,         # graphiques d'entrainement
            save=True,
            save_period=10,     # sauvegarde tous les 10 epochs
            val=True,
            patience=50,        # early stopping
            cache=False,        # pas de cache pour eviter les problemes
            workers=4,          # nombre de workers
            optimizer='AdamW',  # optimiseur
            close_mosaic=10,    # fermer mosaic augmentation
            amp=True,           # mixed precision
            fraction=1.0,       # utiliser 100% des donnees
            profile=False,      # pas de profiling
            freeze=None,        # pas de layers freezes
            multi_scale=False,  # pas de multi-scale pour stabilite
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            augment=True,       # augmentation des donnees
            agnostic_nms=False,
            retina_masks=False,
            show_labels=True,
            show_conf=True,
            show_boxes=True
        )
        
        print(f"\nentrainement termine avec succes !")
        
        # sauvegarder le modele final
        model_dir = script_dir / "models" / "yolov8_accessibility_full"
        best_model = model_dir / "weights" / "best.pt"
        last_model = model_dir / "weights" / "last.pt"
        
        final_path = script_dir / "models" / "yolov8_accessibility_trained.pt"
        
        if best_model.exists():
            import shutil
            shutil.copy2(best_model, final_path)
            print(f"modele final sauvegarde: {final_path}")
            
            # afficher les resultats
            print(f"\nresultats d'entrainement:")
            print(f"  modele: {final_path}")
            print(f"  classes: 5 (step, stair, ramp, handrail, grab_bar)")
            print(f"  epochs: 80")
            print(f"  donnees: 2693 train, 674 val")
            
        else:
            print(f"attention: modele best.pt non trouve dans {model_dir}")
            if last_model.exists():
                import shutil
                shutil.copy2(last_model, final_path)
                print(f"modele last.pt sauvegarde: {final_path}")
        
        return results
        
    except Exception as e:
        print(f"erreur lors de l'entrainement: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = train_full()
    if results:
        print(f"\nsucces ! modele pret pour les tests")
    else:
        print(f"\nechec de l'entrainement")
