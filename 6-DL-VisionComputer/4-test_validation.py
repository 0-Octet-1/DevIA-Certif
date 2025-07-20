import json
import os
import sys
import argparse
import importlib.util
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# import dynamique du modele yolo
script_dir = Path(__file__).parent
spec = importlib.util.spec_from_file_location("modele_yolo", script_dir / "3-modele_yolo.py")
yolo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(yolo_module)
accessibility_yolo = yolo_module.accessibility_yolo

# import du post-processing
from yolo_postprocessing import decode_yolo_predictions, analyze_accessibility_from_detections

def test_single_image(image_path, model_path=None, output_dir=None, confidence_threshold=0.1):
    """teste le modele yolo sur une image specifique"""
    
    # chemins par defaut
    if model_path is None:
        model_path = script_dir / "models" / "yolo_accessibility.pth"
    if output_dir is None:
        output_dir = script_dir / "test_outputs"
    
    # verifier que l'image existe
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"erreur: image non trouvee: {image_path}")
        return False
    
    # charger le modele
    model = accessibility_yolo(num_classes=4)
    if not Path(model_path).exists():
        print(f"erreur: modele non trouve: {model_path}")
        return False
    
    model.load_model(str(model_path))
    print(f"modele yolo charge: {model_path}")
    print(f"test image: {image_path.name}")
    
    # prediction
    result = model.predict(str(image_path))
    
    if result:
        if 'predictions_tensor' in result:
            predictions_tensor = result['predictions_tensor']
            print(f"shape sortie: {predictions_tensor.shape}")
            print(f"min: {predictions_tensor.min():.4f}, max: {predictions_tensor.max():.4f}, mean: {predictions_tensor.mean():.4f}")
            
            # decodage des predictions
            detections = decode_yolo_predictions(predictions_tensor, confidence_threshold=confidence_threshold)
            print(f"detections trouvees: {len(detections)}")
            
            for det in detections:
                print(f"  {det['label']}: conf={det['confidence']:.3f}, bbox={det['bbox']}")
            
            # analyse accessibilite
            analysis = analyze_accessibility_from_detections(detections)
            print(f"accessibilite: {analysis['status']} ({analysis['score']}%)")
            
            # visualisation
            visualize_predictions(str(image_path), detections, output_dir)
            
            return True
        else:
            print(f"prediction: {result.get('message', 'erreur format result')}")
            return False
    else:
        print("aucune prediction obtenue")
        return False

def test_dataset_images(model_path=None, num_images=5, confidence_threshold=0.1):
    """teste le modele sur des images du dataset avec annotations"""
    
    # chemins par defaut
    if model_path is None:
        model_path = script_dir / "models" / "yolo_accessibility.pth"
    
    annotations_path = script_dir / "data" / "parsed_annotations.json"
    if not annotations_path.exists():
        print(f"erreur: annotations non trouvees: {annotations_path}")
        return False
    
    # charger le modele
    model = accessibility_yolo(num_classes=4)
    if not Path(model_path).exists():
        print(f"erreur: modele non trouve: {model_path}")
        return False
    
    model.load_model(str(model_path))
    print(f"modele yolo charge: {model_path}")
    
    # charger les annotations
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', [])
    
    # trouver des images avec annotations
    test_images = []
    for i, annotation in enumerate(annotations[:num_images]):
        if isinstance(annotation, dict) and annotation.get('boxes'):
            img_name = annotation.get('image_name', f'image_{i}.jpg')
            img_path = script_dir / "data" / "images" / img_name
            if img_path.exists():
                test_images.append((img_path, img_name, annotation))
    
    print(f"test sur {len(test_images)} images du dataset")
    
    # tester chaque image
    for i, (img_path, img_name, ground_truth) in enumerate(test_images):
        print(f"\n--- image {i+1}: {img_name} ---")
        
        # ground truth
        boxes = ground_truth['boxes']
        print(f"verite terrain: {len(boxes)} boites")
        for j, box_data in enumerate(boxes):
            label = box_data['label']
            bbox = box_data['bbox_abs']
            print(f"  boite {j+1}: {label} {bbox}")
        
        # prediction
        result = model.predict(str(img_path))
        
        if result and 'predictions_tensor' in result:
            predictions_tensor = result['predictions_tensor']
            detections = decode_yolo_predictions(predictions_tensor, confidence_threshold=confidence_threshold)
            print(f"detections trouvees: {len(detections)}")
            
            for det in detections:
                print(f"  {det['label']}: conf={det['confidence']:.3f}, bbox={det['bbox']}")
            
            analysis = analyze_accessibility_from_detections(detections)
            print(f"accessibilite: {analysis['status']} ({analysis['score']}%)")
            
            # visualisation avec ground truth
            visualize_image_with_boxes(str(img_path), ground_truth, img_name)
    
    return True

def show_training_history(history_path=None):
    """affiche l'historique d'entrainement"""
    if history_path is None:
        history_path = script_dir / "models" / "yolo_accessibility_history.json"
    
    if not Path(history_path).exists():
        print(f"historique non trouve: {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    print(f"\n--- historique entrainement ---")
    print(f"epochs: {history['epochs']}")
    print(f"batch_size: {history['batch_size']}")
    print(f"learning_rate: {history['learning_rate']}")
    
    print(f"\nloss finale:")
    print(f"train: {history['train_loss'][-1]:.6f}")
    print(f"val: {history['val_loss'][-1]:.6f}")

def visualize_predictions(img_path, detections, output_dir):
    """visualise les predictions sur une image"""
    try:
        # charger image
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        
        # creer figure
        plt.figure(figsize=(12, 8))
        plt.imshow(img_array)
        plt.title(f"predictions: {Path(img_path).name}")
        
        # dessiner les boites de detection
        ax = plt.gca()
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            label = detection['label']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # rectangle
            color = colors[i % len(colors)]
            rect = plt.Rectangle((x1, y1), width, height, 
                               fill=False, color=color, linewidth=2)
            ax.add_patch(rect)
            
            # label avec confidence
            plt.text(x1, y1-10, f"{label} ({confidence:.2f})", 
                    color=color, fontsize=10, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        plt.axis('off')
        plt.tight_layout()
        
        # sauvegarder
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        img_name = Path(img_path).stem
        output_path = output_dir / f"predictions_{img_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"predictions sauvees: {output_path}")
        
    except Exception as e:
        print(f"erreur visualisation predictions: {e}")

def visualize_image_with_boxes(img_path, ground_truth, img_name):
    """affiche une image avec ses boites de verite terrain"""
    try:
        # charger image
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        
        # creer figure
        plt.figure(figsize=(10, 8))
        plt.imshow(img_array)
        plt.title(f"ground truth: {img_name}")
        
        # dessiner les boites
        ax = plt.gca()
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, box_data in enumerate(ground_truth['boxes']):
            bbox = box_data['bbox_abs']
            label_name = box_data['label']
            label_id = box_data['label_id']
            
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # rectangle
            rect = plt.Rectangle((x1, y1), width, height, 
                               fill=False, color=colors[label_id % len(colors)], linewidth=2)
            ax.add_patch(rect)
            
            # label
            plt.text(x1, y1-10, f"{label_name}", 
                    color=colors[label_id % len(colors)], fontsize=12, weight='bold')
        
        plt.axis('off')
        plt.tight_layout()
        
        # sauvegarder
        output_dir = script_dir / "test_outputs"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"ground_truth_{img_name.replace('.jpg', '.png')}"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"ground truth sauve: {output_path}")
        
    except Exception as e:
        print(f"erreur visualisation ground truth: {e}")

def main():
    """fonction principale avec arguments en ligne de commande"""
    parser = argparse.ArgumentParser(description="test du modele yolo pour detection accessibilite")
    parser.add_argument("--image", "-i", type=str, help="chemin vers une image specifique a tester")
    parser.add_argument("--model", "-m", type=str, help="chemin vers le modele (defaut: models/yolo_accessibility.pth)")
    parser.add_argument("--output", "-o", type=str, help="dossier de sortie (defaut: test_outputs)")
    parser.add_argument("--confidence", "-c", type=float, default=0.1, help="seuil de confiance (defaut: 0.1)")
    parser.add_argument("--dataset", "-d", action="store_true", help="tester sur des images du dataset")
    parser.add_argument("--num-images", "-n", type=int, default=5, help="nombre d'images du dataset a tester (defaut: 5)")
    parser.add_argument("--history", action="store_true", help="afficher l'historique d'entrainement")
    
    args = parser.parse_args()
    
    print("=== test du modele yolo ===")
    
    # test sur une image specifique
    if args.image:
        print(f"\ntest image specifique: {args.image}")
        success = test_single_image(
            image_path=args.image,
            model_path=args.model,
            output_dir=args.output,
            confidence_threshold=args.confidence
        )
        if not success:
            print("echec du test")
            return 1
    
    # test sur le dataset
    elif args.dataset:
        print(f"\ntest sur {args.num_images} images du dataset")
        success = test_dataset_images(
            model_path=args.model,
            num_images=args.num_images,
            confidence_threshold=args.confidence
        )
        if not success:
            print("echec du test dataset")
            return 1
    
    # par defaut: test rapide sur une image du dataset
    else:
        print("\ntest rapide (utilisez --help pour plus d'options)")
        success = test_dataset_images(
            model_path=args.model,
            num_images=1,
            confidence_threshold=args.confidence
        )
        if not success:
            print("echec du test rapide")
            return 1
    
    # afficher l'historique si demande
    if args.history:
        show_training_history()
    
    print("\ntest termine avec succes")
    return 0

if __name__ == "__main__":
    sys.exit(main())
