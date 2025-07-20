#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug du post-processing yolo
"""

import torch
import numpy as np
from yolo_postprocessing import decode_yolo_predictions

def test_simple():
    """test basique du post-processing"""
    print("=== test post-processing ===")
    
    # creation d'un tensor de test avec des valeurs realistes
    predictions = torch.randn(1, 27, 112, 63) * 0.1
    
    # ajout d'une detection artificielle forte
    # cellule (50, 30), ancre 0
    predictions[0, 0, 50, 30] = 0.5   # x_offset
    predictions[0, 1, 50, 30] = 0.3   # y_offset  
    predictions[0, 2, 50, 30] = 0.2   # w_scale
    predictions[0, 3, 50, 30] = 0.4   # h_scale
    predictions[0, 4, 50, 30] = 3.0   # obj_conf (avant sigmoid = 0.95)
    predictions[0, 5, 50, 30] = 5.0   # class 0 (step) - forte probabilite
    predictions[0, 6, 50, 30] = -2.0  # class 1
    predictions[0, 7, 50, 30] = -2.0  # class 2  
    predictions[0, 8, 50, 30] = -2.0  # class 3
    
    print(f"tensor shape: {predictions.shape}")
    print(f"valeurs test cellule (50,30): {predictions[0, :9, 50, 30]}")
    
    # test avec seuil bas
    detections = decode_yolo_predictions(predictions, confidence_threshold=0.1)
    print(f"detections trouvees (seuil 0.1): {len(detections)}")
    
    for i, det in enumerate(detections):
        print(f"  detection {i+1}: {det}")
    
    # test avec seuil tres bas
    detections_low = decode_yolo_predictions(predictions, confidence_threshold=0.01)
    print(f"detections trouvees (seuil 0.01): {len(detections_low)}")

if __name__ == "__main__":
    test_simple()
