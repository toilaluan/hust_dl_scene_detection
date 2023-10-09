from scene_detect import Scener
from scene_detect.data import HanoiDataset
import argparse
import yaml
from torch.utils.data import DataLoader
import easydict
import torchvision.transforms as T
import pytorch_lightning as pl
import timm
from pytorch_lightning import Trainer


def get_transform(backbone_name):
    try:
        data_config = timm.get_pretrained_cfg(backbone_name).to_dict()
        mean = data_config["mean"]
        std = data_config["std"]
    except:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return transform


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="config.yaml")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    with open(args.cfg_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = easydict.EasyDict(cfg)
        cfg.data.img_size = eval(cfg.data.img_size)
    transforms = get_transform(cfg.model.model_name)
    print(transforms)
    train_ds = HanoiDataset(
        cfg.data.image_root,
        cfg.data.train_annotation_file,
        cfg.data.img_size,
        transforms=transforms,
        is_training=True,
    )
    val_ds = HanoiDataset(
        cfg.data.image_root,
        cfg.data.test_annotation_file,
        cfg.data.img_size,
        transforms=transforms,
        is_training=False,
    )
    print("Total number of train images: ", len(train_ds))
    print("Total number of val images: ", len(val_ds))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = Scener(cfg)
    logger = pl.loggers.TensorBoardLogger(
        save_dir="tensorboard_logs",
    )
    trainer = Trainer(
        gpus=1,
        max_epochs=args.max_epochs,
        logger=logger,
        devices=1,
        log_every_n_steps=5,
    )
    trainer.fit(model, train_loader, val_loader)
