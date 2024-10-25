import numpy as np


class EarlyStopping():
    def __init__(self, patience=3, verbose=False, delta=0, path=path):
 
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False # 조기 종료를 의미하며 초기값은 False로 설정
        self.delta = delta # 오차가 개선되고 있다고 판단하기 위한 최소 변화량
        self.path = path
        self.val_loss_min = np.I
    def __call__(self, val_loss, model):
# 에포크 만큼 한습이 반복되면서 best_loss가 갱신되고, bset_loss에 진전이 없으면 조기종료 후 모델을 저장
        score = -val_loss

        if best_score is None:
            self.bset_score = score
            self.save_checkpoint(val_loss, mode)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'Early Stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = Tr
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        def save_checkpoint(self, val_loss, model):
            if self.verbos:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}. Saving model ...')
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss