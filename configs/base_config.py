import torch

class BaseConfig:
    def __init__(self):
        # 학습 경로
        self.train_data_dir_path = './data/train'
        self.train_data_info_file_path = './data/train.csv'
        self.save_result_path = './train_result'

        # 추론 경로
        self.test_data_dir_path = './data/test'
        self.test_data_info_file_path = './data/test.csv'

        # 데이터 분할
        self.test_size = 0.2

        # 고정값
        self.num_classes = 500
        self.train_shuffle = True
        self.val_shuffle = False
        self.test_shuffle = False

        # 하이퍼 파라미터
        self.batch_size = 64
        self.num_workers = 4
        self.lr = 0.001  # Learning rate
        self.epochs = 5

        # 스케줄러 초기화
        self.scheduler_gamma = 0.1  # 학습률을 현재의 10%로 감소

        # 2 epoch마다 학습률을 감소시키는 스케줄러 선언
        self.epochs_per_lr_decay = 2

        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = BaseConfig()