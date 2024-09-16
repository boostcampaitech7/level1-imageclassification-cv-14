import pandas as pd
import torch.optim as optim

from configs.clip_config import config
from utils.data_related import data_split, get_dataloader
from transforms.clip_transform import ClipProcessor
from dataset.clip_dataset import ClipCustomDataset
from models.clip_model import ClipCustomModel
from losses.clip_loss import CLIPLoss
from trainers.clip_trainer import CLIPTrainer
from utils.inference import inference_clip, load_model


def main():
    train_info = pd.read_csv(config.train_data_info_file_path)

    train_df, val_df = data_split(train_info, config.test_size, train_info['target'])

    train_transform = ClipProcessor(transform_name=config.transform_name)
    val_transform = ClipProcessor(transform_name=config.transform_name)

    train_dataset = ClipCustomDataset(config.train_data_dir_path,
                                  train_df,
                                  train_transform,
                                  is_inference = False,
                                  is_train=True)
    
    val_dataset = ClipCustomDataset(config.train_data_dir_path,
                                  val_df,
                                  val_transform,
                                  is_inference = False)
    
    train_loader = get_dataloader(train_dataset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  shuffle=config.train_shuffle,
                                  collate_fn=train_dataset.preprocess)
    
    val_loader = get_dataloader(val_dataset,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                shuffle=config.val_shuffle,
                                collate_fn=val_dataset.preprocess)
    
    model = ClipCustomModel(config.model_name)

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

    loss_fn = CLIPLoss()

    trainer = CLIPTrainer(
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

    test_transform = ClipProcessor(config.model_name)

    test_dataset = ClipCustomDataset(config.test_data_dir_path,
                                  test_info,
                                  test_transform,
                                  is_inference=True)
    
    test_loader = get_dataloader(test_dataset,
                                 batch_size=config.batch_size,
                                 num_workers=config.num_workers,
                                 shuffle=config.test_shuffle,
                                 drop_last=False)
    
    model = ClipCustomModel(config.model_name)

    model.load_state_dict(
        load_model(config.save_result_path, "best_model.pt")
    )

    predictions = inference_clip(model, 
                            config.device, 
                            test_loader)

    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv("output.csv", index=False)

if __name__ == "__main__":
    main()
    test()