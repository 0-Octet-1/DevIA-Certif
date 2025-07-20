#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug du modele yolo pour comprendre pourquoi il ne detecte rien
"""

import torch
import numpy as np
from PIL import Image
from yolo_inference import accessibility_yolo_inference
import os

def debug_yolo_predictions():
    """debug complet du pipeline yolo"""
    
    # chargement modele
    model_path = "../6-DL-VisionComputer/models/yolo_accessibility.pth"
    model = accessibility_yolo_inference(num_classes=4)
    
    if not model.load_model(model_path):
        print("erreur chargement modele")
        return
    
    # image test (creer une image simple)
    test_image = Image.new('RGB', (224, 224), color='red')
    
    # preprocessing
    image_array = np.array(test_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    
    print(f"input tensor shape: {image_tensor.shape}")
    print(f"input tensor range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    
    # inference
    with torch.no_grad():
        predictions = model.predict(image_tensor)
    
    print(f"\npredictions shape: {predictions.shape}")
    print(f"predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"predictions mean: {predictions.mean():.3f}")
    print(f"predictions std: {predictions.std():.3f}")
    
    # analyse detaillee
    print(f"\npredictions raw (first 10 values):")
    print(predictions.flatten()[:10])
    
    # test avec image vraie
    try:
        # charger image utilisateur si possible
        user_image_path = r"C:\Users\grego\Downloads\Chapeau_230_972_img_9486.jpg"
        if os.path.exists(user_image_path):
            print(f"\n=== test avec image utilisateur ===")
            real_image = Image.open(user_image_path)
            real_image = real_image.convert('RGB').resize((224, 224))
            
            real_array = np.array(real_image).astype(np.float32) / 255.0
            real_tensor = torch.from_numpy(real_array).permute(2, 0, 1).unsqueeze(0)
            
            with torch.no_grad():
                real_predictions = model.predict(real_tensor)
            
            print(f"real image predictions shape: {real_predictions.shape}")
            print(f"real image predictions range: [{real_predictions.min():.3f}, {real_predictions.max():.3f}]")
            print(f"real image predictions mean: {real_predictions.mean():.3f}")
            
            # comparaison
            print(f"\ndifference avec image test:")
            print(f"max diff: {torch.abs(predictions - real_predictions).max():.6f}")
            
    except Exception as e:
        print(f"erreur test image reelle: {e}")
    
    # test architecture modele
    print(f"\n=== architecture modele ===")
    print(f"device: {model.device}")
    print(f"model type: {type(model.model)}")
    
    # test des couches
    for name, param in model.model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}, grad: {param.grad is not None}")
    
    return predictions

if __name__ == "__main__":
    debug_yolo_predictions()
