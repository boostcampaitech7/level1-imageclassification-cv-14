import torch

class EfficientNetConfig:
    def __init__(self):
        # 학습 경로
        self.train_data_dir_path = './data/train'
        self.train_data_info_file_path = './data/train.csv'
        self.save_result_path = './efficientnet_folder/bs32_lr1e4_image_processing'

        # 추론 경로
        self.test_data_dir_path = './data/test'
        self.test_data_info_file_path = './data/test.csv'

        # 출력 파일 명
        self.output_name = './output_result/efficientnet_image_processing.csv'

        # 데이터 분할
        self.test_size = 0.2

        # 고정값
        self.num_classes = 500
        self.train_shuffle = True
        self.val_shuffle = False
        self.test_shuffle = False
        self.cv_shuffle = True

        # 모델이름
        self.model_name = "tf_efficientnet_b5.ns_jft_in1k"
        self.transform_name = "tf_efficientnet_b5.ns_jft_in1k"

        # 하이퍼 파라미터
        self.batch_size = 32
        self.num_workers = 4
        
        self.lr = 1e-4  # Learning rate
        self.epochs = 5
        self.n_splits = 5

        # 스케줄러 초기화
        self.scheduler_gamma = 0.5 

        # 2 epoch마다 학습률을 감소시키는 스케줄러 선언
        self.epochs_per_lr_decay = 1

        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = EfficientNetConfig()