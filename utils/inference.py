import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import mode
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
            # predictions.extend(logits.cpu().detach().numpy())  # 결과를 CPU로 옮기고 리스트에 추가, 확률 추출
            predictions.extend(preds.cpu().detach().numpy()) # target 추출
    
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
    
    del model
    torch.cuda.empty_cache()
    
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

def ensemble_predict(models, dataloader, device, num_classes, inference_func, **kwargs):
    '''
    soft voting 방식의 ensemble
    '''
    predictions = np.zeros((len(dataloader.dataset), num_classes))
    for model in models:
        probs = inference_func(model, device, dataloader, **kwargs)
        predictions += probs

    predictions = predictions / len(models)
    return predictions.argmax(axis=1)


def extrat_probs(models, dataloader, device, num_classes, inference_func, **kwargs):
    '''
    동일한 모델의 cross validation 결과를 각 클래스별 확률로 soft voting한 결과 반환
    '''
    predictions = np.zeros((len(dataloader.dataset), num_classes))
    for model in models:
        probs = inference_func(model, device, dataloader, **kwargs)
        predictions += probs

    predictions = predictions / len(models)
    return predictions

def save_probs(df, pred):
    label = [str(i) for i in range(pred.shape[1])]
    probs_df = pd.DataFrame(pred, columns=label)
    new_df = pd.concat([df, probs_df], axis=1)

    return new_df
    
def csv_soft_voting(inputs, num_classes):
    probs_list = [pd.read_csv(path).iloc[:, 1:] for path in inputs]
    total_probs = np.zeros((probs_list[0].shape[0], num_classes))

    for df in probs_list:
        total_probs += df
    total_probs /= len(probs_list)
    return total_probs.to_numpy().argmax(axis=1)

def csv_hard_voting(inputs):
    target_list = [pd.read_csv(path).iloc[:, 1:].to_numpy().argmax(axis=1) for path in inputs]
    total_target = np.stack(target_list, axis=1)

    return mode(total_target, axis=1)[0]

def csv_weighted_voting(inputs, num_classes):
    probs_list = [pd.read_csv(path).iloc[:, 1:] for path in inputs]
    total_probs = np.zeros((probs_list[0].shape[0], num_classes))

    inputs_dict = {k : v for k, v in zip(inputs, probs_list)}

    for k in inputs_dict:
        tmp = int(k[-5:-4])
        total_probs += (inputs_dict[k] * tmp)
    return total_probs.to_numpy().argmax(axis=1)
