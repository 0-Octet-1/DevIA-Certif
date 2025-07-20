import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import numpy as np

class accessibility_detection_dataset(Dataset):
    """dataset pytorch pour detection d'objets accessibilite"""
    
    def __init__(self, annotations_path, transform=None, min_box_area=100):
        """
        args:
            annotations_path: chemin vers le json des annotations parsees
            transform: transformations d'augmentation
            min_box_area: aire minimale des boites (filtre les trop petites)
        """
        with open(annotations_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.annotations = data['annotations']
        self.labels_map = data['labels_map']
        self.transform = transform
        self.min_box_area = min_box_area
        
        # filtrer les images avec au moins une boite valide
        self.valid_annotations = []
        for ann in self.annotations:
            valid_boxes = self._filter_valid_boxes(ann['boxes'], ann['width'], ann['height'])
            if valid_boxes:
                ann_copy = ann.copy()
                ann_copy['boxes'] = valid_boxes
                self.valid_annotations.append(ann_copy)
        
        print(f"dataset charge: {len(self.valid_annotations)} images valides")
    
    def __len__(self):
        return len(self.valid_annotations)
    
    def __getitem__(self, idx):
        """retourne image et targets pour l'entrainement"""
        try:
            ann = self.valid_annotations[idx]
            
            # charger image avec gestion d'erreur
            try:
                image = Image.open(ann['image_path']).convert('RGB')
            except Exception as e:
                print(f"erreur chargement image {ann['image_path']}: {e}")
                # image de remplacement
                image = Image.new('RGB', (224, 224), color='gray')
            
            # preparer targets
            boxes = []
            labels = []
            attributes = []
            
            for box in ann['boxes']:
                # verifier validite de la boite
                bbox = box['bbox_abs']
                if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    boxes.append(bbox)
                    labels.append(box['label_id'])
                    attributes.append(box['attributes'])
            
            # si aucune boite valide, creer une boite factice
            if not boxes:
                boxes = [[10, 10, 50, 50]]
                labels = [1]  # step par defaut
                attributes = [{}]
            
            # convertir en tenseurs
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
            # calculer areas
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            
            # targets pour faster r-cnn
            target = {
                'boxes': boxes,
                'labels': labels,
                'area': areas,
                'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'attributes': attributes
            }
            
            # appliquer transformations
            if self.transform:
                image = self.transform(image)
            
            return image, target
            
        except Exception as e:
            print(f"erreur dataset idx {idx}: {e}")
            # retourner un sample factice en cas d'erreur
            dummy_image = torch.zeros(3, 224, 224)
            dummy_target = {
                'boxes': torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                'labels': torch.tensor([1], dtype=torch.int64),
                'area': torch.tensor([1600.0]),
                'iscrowd': torch.tensor([0], dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'attributes': [{}]
            }
            return dummy_image, dummy_target
    
    def _filter_valid_boxes(self, boxes, img_width, img_height):
        """filtre les boites trop petites ou invalides"""
        valid_boxes = []
        
        for box in boxes:
            x1, y1, x2, y2 = box['bbox_abs']
            
            # verifier coordonnees valides
            if x2 <= x1 or y2 <= y1:
                continue
                
            # verifier dans les limites de l'image
            if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
                continue
                
            # verifier aire minimale
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_box_area:
                continue
                
            valid_boxes.append(box)
        
        return valid_boxes
    
    def get_class_weights(self):
        """calcule les poids des classes pour equilibrage"""
        label_counts = torch.zeros(len(self.labels_map))
        
        for ann in self.valid_annotations:
            for box in ann['boxes']:
                label_counts[box['label_id']] += 1
        
        # poids inverse de la frequence
        total = label_counts.sum()
        weights = total / (len(self.labels_map) * label_counts)
        weights[label_counts == 0] = 0  # eviter division par zero
        
        return weights
    
    def get_stats(self):
        """statistiques du dataset filtre"""
        stats = {
            'total_images': len(self.valid_annotations),
            'total_boxes': sum(len(ann['boxes']) for ann in self.valid_annotations),
            'labels_count': torch.zeros(len(self.labels_map)),
            'avg_boxes_per_image': 0,
            'image_sizes': []
        }
        
        for ann in self.valid_annotations:
            stats['image_sizes'].append((ann['width'], ann['height']))
            for box in ann['boxes']:
                stats['labels_count'][box['label_id']] += 1
        
        stats['avg_boxes_per_image'] = stats['total_boxes'] / stats['total_images']
        
        return stats

def get_transforms(train=True):
    """transformations pour entrainement et validation"""
    if train:
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def collate_fn(batch):
    """fonction de collation pour dataloader"""
    images, targets = zip(*batch)
    return list(images), list(targets)

def main():
    """test du dataset"""
    dataset = accessibility_detection_dataset(
        annotations_path="data/parsed_annotations.json",
        transform=get_transforms(train=True)
    )
    
    print("statistiques dataset:")
    stats = dataset.get_stats()
    print(f"images: {stats['total_images']}")
    print(f"boites: {stats['total_boxes']}")
    print(f"moyenne boites/image: {stats['avg_boxes_per_image']:.2f}")
    
    print("\nrepartition labels:")
    for label, idx in dataset.labels_map.items():
        count = int(stats['labels_count'][idx])
        print(f"  {label}: {count}")
    
    print("\npoids classes:")
    weights = dataset.get_class_weights()
    for label, idx in dataset.labels_map.items():
        print(f"  {label}: {weights[idx]:.3f}")
    
    # test premier echantillon
    print("\ntest premier echantillon:")
    image, target = dataset[0]
    print(f"image shape: {image.shape}")
    print(f"nombre boites: {len(target['boxes'])}")
    print(f"labels: {target['labels']}")
    print(f"attributs: {target['attributes']}")

if __name__ == "__main__":
    main()
