import torch

class BaseConfig:
    def __init__(self):
        # 학습 경로
        self.train_data_dir_path = '/data/ephemeral/home/level1-imageclassification-cv-14/data/train'
        self.train_data_info_file_path = '/data/ephemeral/home/level1-imageclassification-cv-14/data/train.csv'
        self.save_result_path = '/data/ephemeral/home/level1-imageclassification-cv-14/train_result'

        # 추론 경로
        self.test_data_dir_path = '/data/ephemeral/home/level1-imageclassification-cv-14/data/test'
        self.test_data_info_file_path = '/data/ephemeral/home/level1-imageclassification-cv-14/data/test.csv'

        # 데이터 분할
        self.test_size = 0.2

        # 고정값
        self.num_classes = 500
        self.train_shuffle = True
        self.val_shuffle = False
        self.test_shuffle = False

        # 하이퍼 파라미터
        self.batch_size = 32
        self.num_workers = 4
        self.lr = 1e-4 # Learning rate
        self.backbone_lr = 1e-5
        self.weight_decay = 5e-4
        self.epochs = 10
        # 백본 고정 해제 시점 (고정 에폭)
        self.freeze_backbone_epochs = 3 # 초기에 classifier만 학습

        # 스케줄러 초기화
        self.scheduler_gamma = 0.1  # 학습률을 현재의 10%로 감소

        # 10 epoch마다 학습률을 감소시키는 스케줄러 선언
        self.epochs_per_lr_decay = 10

        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = BaseConfig()