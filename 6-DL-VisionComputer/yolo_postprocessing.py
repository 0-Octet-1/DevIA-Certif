#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
post-processing yolo pour decoder les predictions en detections
adapte a l'architecture du modele custom du dossier 6
"""

import torch
import torch.nn.functional as F
import numpy as np

def decode_yolo_predictions(predictions, confidence_threshold=0.3, nms_threshold=0.5, input_size=224):
    """
    decode les predictions yolo en detections - version simplifiee et robuste
    """
    print(f"debut decode_yolo_predictions: shape={predictions.shape}, seuil={confidence_threshold}")
    
    try:
        batch_size, channels, grid_h, grid_w = predictions.shape
        num_anchors = 3
        num_classes = 4
        
        # labels
        labels = {0: 'step', 1: 'stair', 2: 'grab_bar', 3: 'ramp'}
        
        detections = []
        
        # approche simplifiee : analyser les activations directement
        # reshape pour avoir [batch, anchors, features, h, w]
        pred_reshaped = predictions.view(batch_size, num_anchors, 5 + num_classes, grid_h, grid_w)
        
        # extraire les composants
        obj_scores = torch.sigmoid(pred_reshaped[:, :, 4, :, :])  # objectness
        class_scores = pred_reshaped[:, :, 5:, :, :]  # logits classes
        
        print(f"obj_scores range: {obj_scores.min():.4f} - {obj_scores.max():.4f}")
        
        # trouver les cellules avec objectness > seuil
        obj_mask = obj_scores > confidence_threshold
        
        if obj_mask.sum() == 0:
            print(f"aucune detection avec seuil {confidence_threshold}, essai avec seuil plus bas")
            obj_mask = obj_scores > (confidence_threshold / 10)  # seuil 10x plus bas
        
        print(f"cellules candidates: {obj_mask.sum().item()}")
        
        # pour chaque detection candidate
        for batch_idx in range(batch_size):
            for anchor_idx in range(num_anchors):
                for i in range(grid_h):
                    for j in range(grid_w):
                        if obj_mask[batch_idx, anchor_idx, i, j]:
                            obj_conf = obj_scores[batch_idx, anchor_idx, i, j].item()
                            
                            # classe la plus probable
                            class_logits = class_scores[batch_idx, anchor_idx, :, i, j]
                            class_probs = torch.softmax(class_logits, dim=0)
                            class_conf, class_id = torch.max(class_probs, dim=0)
                            
                            final_conf = obj_conf * class_conf.item()
                            
                            print(f"detection candidate: ({i},{j}) ancre {anchor_idx}, obj={obj_conf:.3f}, class={class_conf.item():.3f}, final={final_conf:.3f}")
                            
                            if final_conf > confidence_threshold:
                                # coordonnees simplifiees
                                x_center = (j + 0.5) / grid_w * input_size
                                y_center = (i + 0.5) / grid_h * input_size
                                box_size = min(input_size / grid_w, input_size / grid_h) * (1 + anchor_idx * 0.5)
                                
                                x_min = max(0, int(x_center - box_size/2))
                                y_min = max(0, int(y_center - box_size/2))
                                width = min(input_size - x_min, int(box_size))
                                height = min(input_size - y_min, int(box_size))
                                
                                detection = {
                                    'label': labels[class_id.item()],
                                    'class_id': class_id.item(),
                                    'confidence': final_conf,
                                    'bbox': [x_min, y_min, width, height]
                                }
                                
                                detections.append(detection)
                                print(f"detection ajoutee: {detection}")
        
        print(f"total detections: {len(detections)}")
        return detections
        
    except Exception as e:
        print(f"erreur dans decode_yolo_predictions: {e}")
        return []

def apply_nms(detections, nms_threshold=0.5):
    """
    non-maximum suppression pour eliminer les detections redondantes
    """
    if len(detections) == 0:
        return []
    
    # calcul des aires des boites
    for det in detections:
        x, y, w, h = det['bbox']
        det['area'] = w * h
    
    # tri par confiance decroissante (deja fait normalement)
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    
    while detections:
        # garder la detection avec la plus haute confiance
        current = detections.pop(0)
        keep.append(current)
        
        # calculer iou avec les detections restantes
        remaining = []
        for det in detections:
            iou = calculate_iou(current['bbox'], det['bbox'])
            if iou < nms_threshold:  # garder si iou faible (pas de chevauchement)
                remaining.append(det)
        
        detections = remaining
    
    return keep

def calculate_iou(box1, box2):
    """
    calcul de l'intersection over union entre deux boites
    format boite: [x_min, y_min, width, height]
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # coordonnees des coins
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2
    
    # intersection
    inter_x_min = max(x1, x2)
    inter_y_min = max(y1, y2)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area

def analyze_accessibility_from_detections(detections):
    """
    analyse l'accessibilite basee sur les objets detectes
    """
    if not detections:
        return {
            'score': 100,
            'status': 'accessible',
            'obstacles': [],
            'recommendations': []
        }
    
    obstacles = []
    accessibility_score = 100
    recommendations = []
    
    for det in detections:
        label = det['label']
        confidence = det['confidence']
        
        if label in ['step', 'stair']:
            # marches et escaliers reduisent l'accessibilite
            penalty = int(confidence * 30)  # jusqu'a -30 points
            accessibility_score -= penalty
            obstacles.append({
                'type': label,
                'confidence': confidence,
                'impact': f'-{penalty} points'
            })
            
            if label == 'step':
                recommendations.append("installer une rampe d'acces")
            else:
                recommendations.append("prevoir un ascenseur ou rampe alternative")
        
        elif label == 'grab_bar':
            # barres d'appui ameliorent l'accessibilite
            bonus = int(confidence * 10)  # jusqu'a +10 points
            accessibility_score += bonus
            obstacles.append({
                'type': label,
                'confidence': confidence,
                'impact': f'+{bonus} points (aide)'
            })
        
        elif label == 'ramp':
            # rampes ameliorent l'accessibilite
            bonus = int(confidence * 15)  # jusqu'a +15 points
            accessibility_score += bonus
            obstacles.append({
                'type': label,
                'confidence': confidence,
                'impact': f'+{bonus} points (acces facilite)'
            })
    
    # borner le score entre 0 et 100
    accessibility_score = max(0, min(100, accessibility_score))
    
    # determination du statut
    if accessibility_score >= 80:
        status = 'accessible'
    elif accessibility_score >= 50:
        status = 'partiellement accessible'
    else:
        status = 'non accessible'
    
    return {
        'score': accessibility_score,
        'status': status,
        'obstacles': obstacles,
        'recommendations': recommendations,
        'num_detections': len(detections)
    }

if __name__ == "__main__":
    # test avec des predictions factices
    print("test du post-processing yolo")
    
    # simulation d'une prediction
    fake_predictions = torch.randn(1, 27, 112, 63) * 0.1
    
    # ajout de quelques "vraies" detections dans le tensor
    # cellule (50, 30) avec une detection step
    fake_predictions[0, 0, 50, 30] = 0.5   # x_offset
    fake_predictions[0, 1, 50, 30] = 0.3   # y_offset  
    fake_predictions[0, 2, 50, 30] = 0.2   # w_scale
    fake_predictions[0, 3, 50, 30] = 0.4   # h_scale
    fake_predictions[0, 4, 50, 30] = 2.0   # obj_conf (avant sigmoid)
    fake_predictions[0, 5, 50, 30] = 3.0   # class 0 (step)
    
    detections = decode_yolo_predictions(fake_predictions, confidence_threshold=0.1)
    print(f"detections trouvees: {len(detections)}")
    
    for det in detections:
        print(f"  {det['label']}: conf={det['confidence']:.3f}, bbox={det['bbox']}")
    
    analysis = analyze_accessibility_from_detections(detections)
    print(f"analyse accessibilite: {analysis}")
