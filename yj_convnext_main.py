import pandas as pd
import torch.optim as optim
import torch
import gc

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from configs.yj_cv_config import config
from utils.data_related import data_split, get_dataloader
from transforms.yj_cv_transform import AlbumentationsTransform
from transforms.sketch_transform_develop import SketchTransform
from dataset.dataset import CustomDataset
from models.yj_convnext_model import Convnext_Model
from losses.cross_entropy_loss import CrossEntropyLoss
from trainers.yj_cv_trainer import Trainer
from utils.inference import inference, load_model, inference_convnext, ensemble_predict 
from losses.LabelSmoothingCrossEntropy import LabelSmoothingCrossEntropy
from utils.TimeDecorator import TimeDecorator
from sklearn.model_selection import StratifiedKFold
from utils.data_related import data_split, get_dataloader, get_subset


# @TimeDecorator()
# def main():
#     train_info = pd.read_csv(config.train_data_info_file_path)

#     train_transform = AlbumentationsTransform(is_train=True)

#     train_dataset = CustomDataset(config.train_data_dir_path,
#                                   train_info,
#                                   train_transform)
    
#     model = Convnext_Model(model_name = "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320", num_classes = 500, pretrained = True)

#     model.to(config.device)

#     optimizer = optim.Adam(
#         model.parameters(),
#         lr=config.lr
#     )

#     loss_fn = CrossEntropyLoss()

#     trainer = Trainer(
#         model=model,
#         device=config.device,
#         train_dataset=train_dataset,  # 전체 학습 데이터셋
#         val_dataset=train_dataset,  # 검증용으로도 동일한 전체 학습 데이터셋 사용
#         optimizer=optimizer,
#         scheduler=1,
#         loss_fn=loss_fn,
#         epochs=7,
#         result_path=config.save_result_path,
#         n_splits=5,  # K-Fold의 K 값, 예를 들어 5로 설정
#         num_workers=config.num_workers
#         )

#     trainer.train_with_cv()
    
    
@TimeDecorator()
def cv_main():
    train_info = pd.read_csv(config.train_data_info_file_path)

    train_transform = AlbumentationsTransform(is_train=True)

    train_dataset = CustomDataset(config.train_data_dir_path,
                                  train_info,
                                  train_transform,
                                  is_inference = False)
    
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_dataset, train_dataset.targets)):
        print(f"Fold {fold+1}/{config.n_splits}")
        print(len(train_idx))
        print(len(val_idx))
        
        train_subset = get_subset(train_dataset, train_idx)
        val_subset = get_subset(train_dataset, val_idx)
        
        train_loader = get_dataloader(train_subset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      shuffle=config.train_shuffle
                                      )
        
        val_loader = get_dataloader(val_subset,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers,
                                    shuffle=config.val_shuffle
                                    )
    
        model = Convnext_Model(model_name = 'convnext_xxlarge.clip_laion2b_soup_ft_in1k', num_classes = 500, pretrained = True)

        model.to(config.device)

        optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=1e-4
        )
    
        # scheduler_step_size = len(train_loader) * config.epochs_per_lr_decay

        # scheduler = optim.lr_scheduler.StepLR(
        #     optimizer,
        #     step_size=scheduler_step_size,
        #     gamma=config.scheduler_gamma
        # )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',        # validation loss가 최소가 될 때 작동, default='min'
            factor=0.1,        # 학습률을 절반으로 감소
            patience=2,        # 1 epoch 동안 개선이 없으면 학습률 감소
            verbose=True       # 학습률 변경 시 로그 출력
        )

        loss_fn = CrossEntropyLoss()

        trainer = Trainer(
            model=model,
            device=config.device,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            epochs=config.epochs,
            result_path=config.save_result_path,
            num_workers=config.num_workers
            )

        trainer.train(fold=fold + 1)
        print(f"Finished Fold {fold + 1}")
        
        # fold 끝난 후 메모리 정리
        del model, optimizer, scheduler, trainer
        torch.cuda.empty_cache()
        gc.collect()

@TimeDecorator()
def cv_test():
    test_info = pd.read_csv(config.test_data_info_file_path)

    test_transform = AlbumentationsTransform(is_train=False)

    test_dataset = CustomDataset(config.test_data_dir_path,
                                  test_info,
                                  test_transform,
                                  is_inference=True)
    
    test_loader = get_dataloader(test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=config.test_shuffle,
                                 drop_last=False,
                                 num_workers=config.num_workers
                                 )
    
    models = []
    for model_path in os.listdir(config.save_result_path):
        model = Convnext_Model(model_name = 'convnext_xxlarge.clip_laion2b_soup_ft_in1k', num_classes = 500, pretrained = True)
        model.load_state_dict(
            load_model(config.save_result_path, model_path)
        )
        models.append(model)
    
    predictions = ensemble_predict(models, 
                                   test_loader, 
                                   config.device,
                                   config.num_classes,
                                   inference_convnext
                                   )
    
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv("output.csv", index=False) 

if __name__ == "__main__":
    #main()
    cv_main()
    cv_test()