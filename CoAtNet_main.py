import pandas as pd
import torch.optim as optim

from configs.CoAt_config import config
from utils.data_related import data_split, get_dataloader
from transforms.CoAt_albumentations_transform import AlbumentationsTransform
from dataset.dataset import CustomDataset
from models.base_timm_model import TimmModel
from losses.cross_entropy_loss import CrossEntropyLoss
from trainers.CoAt_trainer import Trainer
from utils.inference import inference, load_model


def main():
    train_info = pd.read_csv(config.train_data_info_file_path)

    train_df, val_df = data_split(train_info, config.test_size, train_info['target'])

    train_transform = AlbumentationsTransform(is_train=True)
    val_transform = AlbumentationsTransform(is_train=False)

    train_dataset = CustomDataset(config.train_data_dir_path,
                                  train_df,
                                  train_transform)
    
    val_dataset = CustomDataset(config.train_data_dir_path,
                                  val_df,
                                  val_transform)
    
    train_loader = get_dataloader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=config.train_shuffle)
    
    val_loader = get_dataloader(val_dataset,
                                batch_size=config.batch_size,
                                shuffle=config.val_shuffle)
    
    model = TimmModel('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=True, num_classes=config.num_classes)

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

    trainer = Trainer(
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

def test():
    test_info = pd.read_csv(config.test_data_info_file_path)

    test_transform = AlbumentationsTransform(is_train=False)

    test_dataset = CustomDataset(config.test_data_dir_path,
                                  test_info,
                                  test_transform,
                                  is_inference=True)
    
    test_loader = get_dataloader(test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=config.test_shuffle,
                                 drop_last=False)
    
    model = TimmModel('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=True, num_classes=config.num_classes)

    model.load_state_dict(
        load_model(config.save_result_path, "best_model.pt")
    )

    predictions = inference(model, 
                            config.device, 
                            test_loader)

    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv("output.csv", index=False)

if __name__ == "__main__":
    main()
    test()