import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# 모델 추론을 위한 함수
def inference(
    model: nn.Module, 
    device: torch.device, 
    test_loader: DataLoader
):
    # 모델을 평가 모드로 설정
    model.to(device)
    model.eval()
    
    predictions = []
    with torch.no_grad():  # Gradient 계산을 비활성화
        for images in tqdm(test_loader):
            # 데이터를 같은 장치로 이동
            images = images.to(device)
            
            # 모델을 통해 예측 수행
            logits = model(images)
            logits = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            # 예측 결과 저장
            predictions.extend(preds.cpu().detach().numpy())  # 결과를 CPU로 옮기고 리스트에 추가
    
    return predictions

def inference_clip(
    model: nn.Module, 
    device: torch.device, 
    test_loader: DataLoader,
    label_to_text : dict
):
    # 모델을 평가 모드로 설정
    model.to(device)
    model.eval()
    
    predictions = []
    with torch.no_grad():  # Gradient 계산을 비활성화
        for batch in tqdm(test_loader):            
            # 모델을 통해 예측 수행
            outputs = model(
                pixel_values = batch['pixel_values'].to(device),
                **label_to_text
            )
            probs = outputs.logits_per_image.softmax(dim = 1)
            # preds = probs.argmax(dim=1)
            
            # 예측 결과 저장
            predictions.extend(probs.cpu().detach().numpy())  # 결과를 CPU로 옮기고 리스트에 추가
    
    return predictions

# 모델 추론을 위한 함수
def inference_convnext(
    model: nn.Module, 
    device: torch.device, 
    test_loader: DataLoader
):
    # 모델을 평가 모드로 설정
    model.to(device)
    model.eval()
    
    predictions = []
    with torch.no_grad():  # Gradient 계산을 비활성화
        for images in tqdm(test_loader):
            # 데이터를 같은 장치로 이동
            if len(images.shape) == 5:  # (num_channels, height, width) 형식인 경우
                images = images.squeeze(1)
            images = images.to(device)
            
            # 모델을 통해 예측 수행
            logits = model(images)
            logits = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            # 예측 결과 저장
            predictions.extend(preds.cpu().detach().numpy())  # 결과를 CPU로 옮기고 리스트에 추가
    
    return predictions

def load_model(path, name):
    return torch.load(os.path.join(path, name), map_location='cpu')

def ensemble_predict(models, dataloader, device, inference_func, num_classes, **kwargs):
    '''
    soft voting 방식의 ensemble
    '''
    predictions = np.zeros((len(dataloader.dataset), num_classes))
    for model in models:
        probs = inference_func(model, device, dataloader, **kwargs)
        predictions += probs

    return predictions / len(models)
