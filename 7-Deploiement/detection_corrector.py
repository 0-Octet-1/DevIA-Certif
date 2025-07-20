#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
correcteur de detections pour ameliorer la pertinence accessibilite
"""

import numpy as np
from typing import List, Dict, Any

def analyze_geometry(bbox: List[float]) -> Dict[str, float]:
    """analyse la geometrie d'une detection"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # ratios geometriques
    aspect_ratio = width / height if height > 0 else 0
    area = width * height
    
    return {
        'width': width,
        'height': height,
        'aspect_ratio': aspect_ratio,
        'area': area,
        'is_horizontal': aspect_ratio > 2.0,  # tres large = rampe potentielle
        'is_large': area > 10000,  # grande surface
        'is_thin': height < 50 and width > 100  # mince et long = rampe
    }

def correct_step_to_ramp(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """corrige les detections step en ramp selon la geometrie"""
    
    corrected = []
    
    for detection in detections:
        corrected_detection = detection.copy()
        
        if detection['class'] == 'step':
            geometry = analyze_geometry(detection['bbox'])
            
            # criteres pour identifier une rampe mal classee
            is_likely_ramp = (
                geometry['is_horizontal'] and  # tres large
                geometry['is_thin'] and        # mince
                geometry['area'] > 5000        # assez grande
            )
            
            if is_likely_ramp:
                corrected_detection['class'] = 'ramp'
                corrected_detection['original_class'] = 'step'
                corrected_detection['correction_applied'] = True
                corrected_detection['correction_reason'] = 'geometrie_rampe'
                
                # ajuster confiance (leger bonus pour correction)
                corrected_detection['confidence'] = min(detection['confidence'] * 1.1, 1.0)
        
        corrected.append(corrected_detection)
    
    return corrected

def group_similar_detections(detections: List[Dict[str, Any]], 
                           distance_threshold: float = 50.0) -> List[Dict[str, Any]]:
    """groupe les detections similaires proches"""
    
    if not detections:
        return detections
    
    grouped = []
    processed = set()
    
    for i, detection in enumerate(detections):
        if i in processed:
            continue
            
        # chercher detections similaires proches
        group = [detection]
        bbox1 = detection['bbox']
        x1_center = (bbox1[0] + bbox1[2]) / 2
        y1_center = (bbox1[1] + bbox1[3]) / 2
        
        for j, other_detection in enumerate(detections[i+1:], i+1):
            if j in processed:
                continue
                
            if other_detection['class'] == detection['class']:
                bbox2 = other_detection['bbox']
                x2_center = (bbox2[0] + bbox2[2]) / 2
                y2_center = (bbox2[1] + bbox2[3]) / 2
                
                distance = np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)
                
                if distance < distance_threshold:
                    group.append(other_detection)
                    processed.add(j)
        
        # si groupe de steps proches, potentiellement un escalier
        if len(group) > 3 and detection['class'] == 'step':
            # creer detection escalier globale
            all_bboxes = [d['bbox'] for d in group]
            min_x = min(bbox[0] for bbox in all_bboxes)
            min_y = min(bbox[1] for bbox in all_bboxes)
            max_x = max(bbox[2] for bbox in all_bboxes)
            max_y = max(bbox[3] for bbox in all_bboxes)
            
            avg_confidence = sum(d['confidence'] for d in group) / len(group)
            
            stair_detection = {
                'class': 'stair',
                'confidence': avg_confidence,
                'bbox': [min_x, min_y, max_x, max_y],
                'grouped_from': len(group),
                'correction_applied': True,
                'correction_reason': 'groupement_steps'
            }
            
            grouped.append(stair_detection)
        else:
            grouped.extend(group)
        
        processed.add(i)
    
    return grouped

def enhance_accessibility_analysis(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """analyse amelioree de l'accessibilite avec logique contextuelle"""
    
    if not detections:
        return {
            "score_accessibilite": 0.8,
            "niveau": "accessible",
            "elements_detectes": [],
            "recommandations": ["pas d'obstacle detecte - espace accessible"],
            "corrections_appliquees": [],
            "nombre_corrections": 0,
            "statistiques": {},
            "analyse_contextuelle": {
                "score_base": 0.8,
                "elements_positifs": 0,
                "elements_negatifs": 0,
                "compensation_detectee": False
            }
        }
    
    # comptage par type
    counts = {}
    corrections = []
    elements = []
    total_confidence = 0
    
    for detection in detections:
        class_name = detection['class']
        confidence = detection['confidence']
        
        counts[class_name] = counts.get(class_name, 0) + 1
        total_confidence += confidence
        
        elements.append({
            "type": class_name,
            "confiance": round(confidence, 3),
            "position": detection['bbox'],
            "corrige": detection.get('correction_applied', False)
        })
        
        if detection.get('correction_applied'):
            corrections.append({
                "original": detection.get('original_class', 'inconnu'),
                "corrige": class_name,
                "raison": detection.get('correction_reason', 'inconnu')
            })
    
    # analyse contextuelle intelligente
    step_count = counts.get('step', 0)
    stair_count = counts.get('stair', 0)
    ramp_count = counts.get('ramp', 0)
    handrail_count = counts.get('handrail', 0)
    grab_bar_count = counts.get('grab_bar', 0)
    
    # logique de scoring contextuelle
    # si aucun obstacle (step/stair), commencer avec un bon score
    if step_count == 0 and stair_count == 0:
        score_base = 0.7  # score positif de base sans obstacles
    else:
        score_base = 0.5  # score neutre si obstacles detectes
    
    # bonus pour elements positifs d'accessibilite
    if ramp_count > 0:
        score_base += 0.2  # rampe = tres bon
    if handrail_count > 0 or grab_bar_count > 0:
        score_base += 0.15  # elements de soutien = bon
    
    # penalite contextuelle pour steps/stairs
    if step_count > 0 or stair_count > 0:
        # penalite reduite si elements d'accessibilite presents
        if ramp_count > 0 or handrail_count > 0:
            score_base -= 0.1  # penalite legere si compensation
        else:
            score_base -= 0.25  # penalite plus forte si pas de compensation
    
    # bonus pour diversite (elements multiples)
    if len(counts) > 1:
        score_base += 0.1
    
    # bonus confiance moyenne
    avg_confidence = total_confidence / len(detections)
    score_base += avg_confidence * 0.2
    
    score_final = max(0.0, min(1.0, score_base))
    
    # determination niveau avec seuils ajustes
    if score_final >= 0.75:
        niveau = "tres accessible"
    elif score_final >= 0.55:
        niveau = "bien accessible"
    elif score_final >= 0.35:
        niveau = "partiellement accessible"
    else:
        niveau = "peu accessible"
    
    # recommandations contextuelles
    recommandations = []
    
    # cas ideal : aucun obstacle detecte
    if step_count == 0 and stair_count == 0:
        if ramp_count > 0:
            recommandations.append("espace accessible avec rampe detectee - configuration ideale")
        elif handrail_count > 0 or grab_bar_count > 0:
            recommandations.append("espace accessible avec elements de soutien - tres bon")
        else:
            recommandations.append("aucun obstacle detecte - acces libre pour pmr")
    
    # cas avec obstacles
    else:
        if ramp_count > 0:
            recommandations.append("rampe d'acces detectee - excellent pour accessibilite")
        
        if handrail_count > 0 or grab_bar_count > 0:
            recommandations.append("elements de soutien detectes - securite renforcee")
        
        if stair_count > 0 and ramp_count == 0:
            recommandations.append("escalier sans rampe alternative - envisager acces pmr")
        elif stair_count > 0 and ramp_count > 0:
            recommandations.append("escalier avec elements d'accessibilite - configuration mixte")
        
        if step_count > 0 and step_count <= 2 and (ramp_count > 0 or handrail_count > 0):
            recommandations.append("quelques marches avec elements d'accessibilite - acceptable")
        elif step_count > 5:
            recommandations.append("nombreuses marches - attention accessibilite")
    
    if not recommandations:
        recommandations.append("elements d'accessibilite detectes")
    
    return {
        "score_accessibilite": round(score_final, 3),
        "niveau": niveau,
        "elements_detectes": elements,
        "recommandations": recommandations,
        "statistiques": counts,
        "corrections_appliquees": corrections,
        "nombre_corrections": len(corrections),
        "analyse_contextuelle": {
            "score_base": round(score_base, 3),
            "elements_positifs": ramp_count + handrail_count + grab_bar_count,
            "elements_negatifs": step_count + stair_count,
            "compensation_detectee": (step_count > 0 or stair_count > 0) and (ramp_count > 0 or handrail_count > 0)
        }
    }

def apply_detection_corrections(detections: List[Dict[str, Any]]) -> tuple:
    """applique toutes les corrections aux detections"""
    
    # etape 1: corrections geometriques
    corrected_detections = correct_step_to_ramp(detections)
    
    # etape 2: groupement intelligent
    grouped_detections = group_similar_detections(corrected_detections)
    
    # etape 3: analyse amelioree
    enhanced_analysis = enhance_accessibility_analysis(grouped_detections)
    
    return grouped_detections, enhanced_analysis

class DetectionCorrector:
    """classe pour corriger et analyser les detections d'accessibilite"""
    
    def __init__(self):
        """initialisation du correcteur"""
        pass
    
    def correct_detections(self, detections: List[Dict[str, Any]], image_shape: tuple) -> tuple:
        """
        corrige les detections et retourne les corrections appliquees
        
        args:
            detections: liste des detections brutes
            image_shape: (height, width) de l'image
            
        returns:
            tuple: (detections_corrigees, liste_corrections)
        """
        # conversion du format pour compatibilite
        formatted_detections = []
        for det in detections:
            formatted_det = {
                'class': det['type'],
                'confidence': det['confiance'],
                'bbox': det['position']
            }
            formatted_detections.append(formatted_det)
        
        # application des corrections
        corrected_detections, analysis = apply_detection_corrections(formatted_detections)
        
        # reconversion au format original
        final_detections = []
        corrections_list = []
        
        for det in corrected_detections:
            final_det = {
                'type': det['class'],
                'confiance': det['confidence'],
                'position': det['bbox'],
                'corrige': det.get('correction_applied', False)
            }
            final_detections.append(final_det)
            
            # ajout des corrections si presentes
            if det.get('correction_applied', False):
                corrections_list.append({
                    'original': det.get('original_class', 'unknown'),
                    'corrige': det['class'],
                    'raison': det.get('correction_reason', 'correction geometrique')
                })
        
        return final_detections, corrections_list
    
    def analyze_accessibility(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        analyse l'accessibilite basee sur les detections
        
        args:
            detections: liste des detections corrigees
            
        returns:
            dict: analyse complete d'accessibilite
        """
        # conversion du format
        formatted_detections = []
        for det in detections:
            formatted_det = {
                'class': det['type'],
                'confidence': det['confiance'],
                'bbox': det['position']
            }
            formatted_detections.append(formatted_det)
        
        # analyse amelioree
        analysis = enhance_accessibility_analysis(formatted_detections)
        
        return analysis
