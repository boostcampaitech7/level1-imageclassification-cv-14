import pandas as pd
import torch
import torch.optim as optim

from configs.CoAtNet_5_config import config
from utils.data_related import data_split, get_dataloader
# from transforms.albumentations_transform import AlbumentationsTransform
from dataset.dataset import CustomDataset
from models.base_timm_model import TimmModel
from models.ViT import ViTModel
from losses.cross_entropy_loss import CrossEntropyLoss
from trainers.CoAtNet_5_trainer import Trainer
from utils.inference import inference, load_model
from transforms.vit_transform import ViTAutoImageTransform


def main():
    train_info = pd.read_csv(config.train_data_info_file_path)

    train_df, val_df = data_split(
        train_info, config.test_size, train_info['target'])

    # train_transform = AlbumentationsTransform(is_train=True)
    # val_transform = AlbumentationsTransform(is_train=False)
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
                                  num_workers=config.num_workers,
                                  shuffle=config.train_shuffle)

    val_loader = get_dataloader(val_dataset,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                shuffle=config.val_shuffle)

    model = ViTModel('google/vit-base-patch16-224', config.num_classes)

    model.to(config.device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay)

    # 스케줄러 설정
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs // 2)

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
        freeze_backbone_epochs=config.freeze_backbone_epochs  # 백본 고정 해제 에폭 설정
    )

    trainer.train()


def test():
    test_info = pd.read_csv(config.test_data_info_file_path)

    # test_transform = AlbumentationsTransform(is_train=False)
    test_transform = ViTAutoImageTransform()

    test_dataset = CustomDataset(config.test_data_dir_path,
                                 test_info,
                                 test_transform,
                                 is_inference=True)

    test_loader = get_dataloader(test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=config.test_shuffle,
                                 num_workers=config.num_workers,
                                 drop_last=False)

    model = ViTModel('google/vit-base-patch16-224', config.num_classes)

    model.load_state_dict(
        load_model(config.save_result_path, "best_model.pt")
    )

    predictions = inference(model,
                            config.device,
                            test_loader)

    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv("output-4.csv", index=False)


if __name__ == "__main__":
    main()
    test()
