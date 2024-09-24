import os

import gc
import torch
import pandas as pd
import torch.optim as optim


from configs.base_config import config
from utils.data_related import data_split, get_dataloader, get_subset
from dataset.dataset import CustomDataset
from models.ViT import ViTModel
from transforms.vit_transform import ViTAutoImageTransform
from losses.cross_entropy_loss import CrossEntropyLoss
from trainers.vit_trainer import ViTTrainer
from utils.inference import inference_vit, load_model, ensemble_predict
from utils.TimeDecorator import TimeDecorator
from sklearn.model_selection import StratifiedKFold

def main():
    train_info = pd.read_csv(config.train_data_info_file_path)
    train_df, val_df = data_split(train_info, config.test_size, train_info['target'])

    train_transform = ViTAutoImageTransform()
    val_transform = ViTAutoImageTransform()

    train_dataset = CustomDataset(config.train_data_dir_path,
                                  train_df,
                                  train_transform)

    val_dataset = CustomDataset(config.train_data_dir_path,
                                val_df,
                                val_transform)

    train_loader = get_dataloader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers = config.num_workers,
                                  shuffle=config.train_shuffle,
                                  )

    val_loader = get_dataloader(val_dataset,
                                batch_size=config.batch_size,
                                num_workers = config.num_workers,
                                shuffle=config.val_shuffle,
                                )
    
    model = ViTModel('google/vit-large-patch16-384', config.num_classes)


    model.to(config.device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=0.1
    )

    scheduler_step_size = len(train_loader) * config.epochs_per_lr_decay

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=config.scheduler_gamma
    )

    loss_fn = CrossEntropyLoss()

    trainer = ViTTrainer(
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

    trainer.train()

def cv_main():
    data_info = pd.read_csv(config.train_data_info_file_path)

    total_transform = ViTAutoImageTransform()

    total_dataset = CustomDataset(config.train_data_dir_path,
                                  data_info,
                                  total_transform,
                                  is_inference = False)
    

    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=config.cv_shuffle)

    for fold, (train_idx, val_idx) in enumerate(skf.split(total_dataset, total_dataset.targets)):
        print(f"Fold {fold+1}/{config.n_splits}")

        train_dataset = get_subset(total_dataset, train_idx)
        val_dataset = get_subset(total_dataset, val_idx)

        train_loader = get_dataloader(train_dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      shuffle=config.train_shuffle,
                                      )
        
        val_loader = get_dataloader(val_dataset,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers,
                                    shuffle=config.val_shuffle,
                                    )
        
        model = ViTModel('google/vit-large-patch16-384', config.num_classes)

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

        trainer = ViTTrainer(
            model=model,
            device=config.device,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            epochs=config.epochs,
            result_path=config.save_result_path,
        )

        trainer.train(fold=fold + 1)
        print(f"Finished Fold {fold + 1}")

        # fold 끝난 후 메모리 정리
        del model, optimizer, scheduler, trainer
        torch.cuda.empty_cache()
        gc.collect()


def cv_test():
    test_info = pd.read_csv(config.test_data_info_file_path)

    test_transform = ViTAutoImageTransform()

    test_dataset = CustomDataset(config.test_data_dir_path,
                                  test_info,
                                  test_transform,
                                  is_inference=True)
    
    test_loader = get_dataloader(test_dataset,
                                 batch_size=config.batch_size,
                                 num_workers=config.num_workers,
                                 shuffle=config.test_shuffle,
                                 drop_last=False,
                                 )
    
    models = []
    for model_path in os.listdir(config.save_result_path):
        print("model path : ", model_path)
        model = ViTModel('google/vit-large-patch16-384', config.num_classes)
        model.load_state_dict(
            load_model(config.save_result_path, model_path)
        )
        models.append(model)
    
    predictions = ensemble_predict(models, 
                                   test_loader, 
                                   config.device,
                                   config.num_classes,
                                   inference_vit,
                                   )
    
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv(config.output_name, index=False)
    


def test():
    test_info = pd.read_csv(config.test_data_info_file_path)

    test_transform = ViTAutoImageTransform()

    test_dataset = CustomDataset(config.test_data_dir_path,
                                    test_info,
                                    test_transform,
                                    is_inference=True)
    
    test_loader = get_dataloader(test_dataset,
                                    batch_size=config.batch_size,
                                    num_workers=config.num_workers,
                                    shuffle=config.test_shuffle,
                                    drop_last=False,
                                )
    
    model = ViTModel('google/vit-large-patch16-384', config.num_classes)

    model.load_state_dict(
        load_model(config.save_result_path, "best_model.pt")
    )

    predictions = inference_vit(model, 
                            config.device, 
                            test_loader
                            )

    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv(config.output_name, index=False)

if __name__ == "__main__":
    # main()
    # cv_main()
    # test()
    cv_test()