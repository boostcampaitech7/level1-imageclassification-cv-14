import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Focal Loss 함수.
        :param gamma: 초점 조정 파라미터, 기본값은 2.0
        :param alpha: 클래스 가중치, 클래스 불균형이 있을 때 사용
        :param reduction: 손실을 합치는 방법 ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Cross-Entropy Loss 계산
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        # 확률로 변환 (softmax)
        pt = torch.exp(-BCE_loss)
        # Focal Loss 계산
        focal_loss = (1 - pt) ** self.gamma * BCE_loss
        
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = at * focal_loss

        # 손실을 합치는 방법에 따라 결과를 반환
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
