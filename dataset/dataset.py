import os
from typing import Tuple, Callable, Union

import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        info_df: pd.DataFrame, 
        transform: Callable,
        is_inference: bool = False
    ):
        # 데이터셋의 기본 경로, 이미지 변환 방법, 이미지 경로 및 레이블을 초기화합니다.
        self.root_dir = root_dir  # 이미지 파일들이 저장된 기본 디렉토리
        self.transform = transform  # 이미지에 적용될 변환 처리
        self.is_inference = is_inference  # 추론인지 확인
        self.image_paths = info_df['image_path'].tolist()  # 이미지 파일 경로 목록
        
        if not self.is_inference:
            self.targets = info_df['target'].tolist()  # 각 이미지에 대한 레이블 목록

    def __len__(self) -> int:
        # 데이터셋의 총 이미지 수를 반환합니다.
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        # 이미지를 읽고 변환하는 부분
        while True:
            try:
                img_path = os.path.join(self.root_dir, self.image_paths[index])  # 이미지 경로 조합
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # 이미지를 BGR 컬러 포맷의 numpy array로 읽어옵니다.

                # 이미지가 손상된 경우 cv2.imread()가 None을 반환할 수 있습니다.
                if image is None:
                    raise ValueError(f"Failed to load image at {img_path}")

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR 포맷을 RGB 포맷으로 변환합니다.
                image = self.transform(image)  # 설정된 이미지 변환을 적용합니다.

                if self.is_inference:
                    return image
                else:
                    target = self.targets[index]  # 해당 이미지의 레이블
                    return image, target  # 변환된 이미지와 레이블을 튜플 형태로 반환합니다.

            except Exception as e:
                # 손상된 파일에 대한 예외 처리
                print(f"Error loading image at index {index}, path {self.image_paths[index]}: {e}")
                
                # 인덱스를 다음으로 넘겨서 다시 시도
                index = (index + 1) % len(self.image_paths)
