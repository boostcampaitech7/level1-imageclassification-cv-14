import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import torch
import pandas as pd
import torch.optim as optim


from configs.efficientnet_config import config
from utils.data_related import get_dataloader, get_subset
from transforms.efficientnet_transform import EfficientNetTransform
from dataset.dataset import CustomDataset
from models.base_timm_model import TimmModel
from losses.cross_entropy_loss import CrossEntropyLoss
from trainers.efficient_trainer import EfficientNetTrainer
from utils.inference import inference, load_model, ensemble_predict
from utils.TimeDecorator import TimeDecorator
from sklearn.model_selection import StratifiedKFold


@TimeDecorator()
def cv_main():
    data_info = pd.read_csv(config.train_data_info_file_path)
    
    train_transform = EfficientNetTransform(is_train=True)
    val_transform = EfficientNetTransform(is_train=False)

    train_dataset = CustomDataset(config.train_data_dir_path,
                                  data_info,
                                  train_transform,
                                  is_inference = False)
    
    val_dataset = CustomDataset(config.train_data_dir_path,
                                data_info,
                                val_transform,
                                is_inference=False)

    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=config.cv_shuffle)

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_dataset, train_dataset.targets)):
        print(f"Fold {fold+1}/{config.n_splits}")

        train_subset_dataset = get_subset(train_dataset, train_idx)
        val_subset_dataset = get_subset(val_dataset, val_idx)

        train_loader = get_dataloader(train_subset_dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      shuffle=config.train_shuffle)
        
        val_loader = get_dataloader(val_subset_dataset,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers,
                                    shuffle=config.val_shuffle)
        
        model = TimmModel(config.model_name,
                          num_classes=config.num_classes,
                          pretrained=True)

        model.to(config.device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr
        )

        scheduler_step_size = len(train_loader) * config.epochs_per_lr_decay

        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_step_size,
            gamma=config.scheduler_gamma
        )

        loss_fn = CrossEntropyLoss()

        trainer = EfficientNetTrainer(
            model=model,
            device=config.device,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            epochs=config.epochs,
            result_path=config.save_result_path
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

    test_transform = EfficientNetTransform(is_train=False)

    test_dataset = CustomDataset(config.test_data_dir_path,
                                  test_info,
                                  test_transform,
                                  is_inference=True)
    
    test_loader = get_dataloader(test_dataset,
                                 batch_size=config.batch_size,
                                 num_workers=config.num_workers,
                                 shuffle=config.test_shuffle,
                                 drop_last=False)
    
    models = []
    for model_path in os.listdir(config.save_result_path):
        model = TimmModel(config.model_name,
                          num_classes=config.num_classes,
                          pretrained=False)
        
        model.load_state_dict(
            load_model(config.save_result_path, model_path)
        )

        models.append(model)
    
    predictions = ensemble_predict(models, 
                                   test_loader, 
                                   config.device,
                                   config.num_classes,
                                   inference)
    
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv(config.output_name, index=False)

if __name__ == "__main__":
    cv_main()
    cv_test()