import itertools
import logging
import os
import sys
from os import listdir
from typing import Optional

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader

from vision.datasets.face_mask import MaskDatasetRetriever
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.data_preprocessing import TestTransform, TrainAugmentation
from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite
from vision.ssd.ssd import MatchPrior
from vision.utils.misc import Timer, freeze_net_layers, store_labels


DEBUG = bool(os.environ.get('DEBUG', False))


def collate_fn(batch):
    return tuple(zip(*batch))


class Config:
    dataset_type = "voc"
    datasets = ""
    validation_dataset = ""
    balance_data = False
    net = "mb3-ssd-lite"
    freeze_base_net = False
    freeze_net = False
    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    gamma = 0.1
    base_net_lr: Optional[float] = None
    extra_layers_lr: Optional[float] = None

    mb2_width_mult = 1.

    # should be initialized
    base_net = None
    pretrained_ssd = None
    resume: Optional[str] = False
    scheduler = "multi-step"
    milestones = "80,100"
    t_max = 120
    batch_size = 32
    num_epochs = 120
    num_workers = 4
    validation_epochs = 5
    debug_steps = 100
    use_cuda = True
    checkpoint_folder = "models/"


# Override
Config.dataset_type = "voc"
# Config.datasets = ["/kaggle/input/facemaskdetection/FaceMaskDataset/"]
Config.datasets = ["/data/datasets/detection/FaceMaskDataset/"]
Config.net = "mb3-ssd-lite"
Config.scheduler = "cosine"
Config.lr = 0.01
Config.t_max = 100
Config.validation_epochs = 5
Config.num_epochs = 100
Config.base_net_lr = 0.001
Config.batch_size = 5
Config.shuffle = False

# Configure
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and Config.use_cuda else "cpu")
if Config.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0

    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = torch.stack(images)
        boxes = torch.stack(boxes)
        labels = torch.stack(labels)

        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        confidence, locations = net(images)

        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = torch.stack(images)
        boxes = torch.stack(boxes)
        labels = torch.stack(labels)

        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device).long()
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
    timer = Timer()
    logging.info(Config)
    assert Config.net == 'mb3-ssd-lite', "The net type is wrong."

    config = mobilenetv1_ssd_config
    config.mode = "light"

    train_transform = TrainAugmentation(config.image_size,
                                        config.image_mean,
                                        config.image_std,
                                        config.mode
                                        )
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")
    datasets = []
    for dataset_path in Config.datasets:
        assert Config.dataset_type == 'voc'
        print(dataset_path)
        dataset = MaskDatasetRetriever(dataset_path + '/train',
                                       transform=train_transform,
                                       target_transform=target_transform,
                                       )

        datasets.append(dataset)

    train_dataset = datasets[0]
    NUM_CLASSES = 3

    if DEBUG:
        logging.info("DEBUG is TRUE. Clipping dataset")
        train_dataset.anno_paths = train_dataset.anno_paths[:1000]

    logging.info("Train dataset size: {}".format(len(train_dataset)))

    train_loader = DataLoader(train_dataset, Config.batch_size,
                              num_workers=Config.num_workers,
                              shuffle=Config.shuffle,
                              drop_last=True,
                              collate_fn=collate_fn,
                              )

    logging.info("Prepare Validation datasets.")
    val_dataset = MaskDatasetRetriever(Config.datasets[0] + '/val',
                                       transform=test_transform,
                                       target_transform=target_transform,
                                       )

    logging.info("validation dataset size: {}".format(len(val_dataset)))
    val_loader = DataLoader(val_dataset, Config.batch_size,
                            num_workers=Config.num_workers,
                            shuffle=False,
                            drop_last=True,
                            collate_fn=collate_fn,
                           )

    logging.info("Build network.")
    net = create_mobilenetv3_ssd_lite(NUM_CLASSES)
    # print(net)

    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = Config.base_net_lr if Config.base_net_lr is not None else Config.lr
    extra_layers_lr = Config.extra_layers_lr if Config.extra_layers_lr is not None else Config.lr
    if Config.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif Config.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    timer.start("Load Model")
    if Config.resume:
        logging.info(f"Resume from the model {Config.resume}")
        net.load(Config.resume)
    elif Config.base_net:
        logging.info(f"Init from base net {Config.base_net}")
        net.init_from_base_net(Config.base_net)
    elif Config.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {Config.pretrained_ssd}")
        net.init_from_pretrained_ssd(Config.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    net.to(DEVICE)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=Config.lr, momentum=Config.momentum,
                                weight_decay=Config.weight_decay)
    logging.info(f"Learning rate: {Config.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    assert Config.scheduler in ('multi-step', 'cosine'), f"Unsupported Scheduler: {Config.scheduler}."
    if Config.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in Config.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                gamma=0.1, last_epoch=last_epoch)
    elif Config.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, Config.t_max, last_epoch=last_epoch)

    logging.info(f"Start training from epoch {last_epoch + 1}.")

    for epoch in range(last_epoch + 1, Config.num_epochs):
        if DEBUG:
            import ipdb; ipdb.set_trace()
        train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=Config.debug_steps, epoch=epoch)
        scheduler.step()  # type: ignore

        if epoch % Config.validation_epochs == 0 or epoch == Config.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            model_path = os.path.join(Config.checkpoint_folder, f"{Config.net}-Epoch-{epoch}-Loss-{val_loss}.pth")
            net.save(model_path)
            logging.info(f"Saved model {model_path}")
