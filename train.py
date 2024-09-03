"""
A PyTorch implemenation of SSD with a ResNet50 backbone
The implementation is modified for custom dataset and to work with newer versions
of Numpy and pycocotools
@original author: Viet Nguyen <nhviet1009@gmail.com>
@modified for custom dataset : Kuljeet Singh 
"""
import os
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '4'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'

import shutil
from argparse import ArgumentParser

import cv2
import torch
import torchvision
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.model import SSD, SSDLite, ResNet, MobileNetV2
from src.utils import generate_dboxes, Encoder, customcoco_classes #CustomCoco here refers to the blood cell types
from src.transform import SSDTransformer
from src.loss import Loss
from src.process import train, evaluate
from src.BCDataset import collate_fn, BCDataset

from apex import amp
from apex.parallel import DistributedDataParallel as ADDP
from torch.nn.parallel import DistributedDataParallel as DDP

DATA_PATH = "customcoco"
ANNOTATION_PATH = "customcoco/annotations"

def check_cuda():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA version:", torch.cuda_version)
        print("CUDA Devices : ", torch.cuda.device_count())
        print(" Setting matmul precision to medium or high to efficiently use RTX-3080")
   #     torch.set_float32_matmul_precision("medium")
    else:
        print("CUDA Unavailable !! Using CPU...")
    return device

def get_args():
    parser = ArgumentParser(description="Implementation of SSD")
    parser.add_argument("--data-path", type=str, default=DATA_PATH,
                        help="the root folder of dataset")
    parser.add_argument("--save-folder", type=str, default="trained_models",
                        help="path to folder containing model checkpoint file")
    parser.add_argument("--log-path", type=str, default="tensorboard/SSD")

    parser.add_argument("--model", type=str, default="ssd", choices=["ssd", "ssdlite"],
                        help="ssd-resnet50 or ssdlite-mobilenetv2")
    parser.add_argument("--epochs", type=int, default=25, help="number of total epochs to run")
    parser.add_argument("--batch-size", type=int, default=8, help="number of samples for each iteration")
    parser.add_argument("--multistep", nargs="*", type=int, default=[43, 54],
                        help="epochs at which to decay learning rate")
    parser.add_argument("--amp", action='store_true', help="Enable mixed precision training")

    parser.add_argument("--lr", type=float, default=2.6e-3, help="initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum argument for SGD optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="momentum argument for SGD optimizer")
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument('--local_rank', default=0, type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using \'python -m multiproc\'.')
    args = parser.parse_args()
    return args

def ddp_setup(opt, rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # if opt.amp :
    #     # initialize the process group
    #     ADDP.init_process_group("gloo", rank=rank, world_size=world_size)
    # else:
    #     DDP.init_process_group("gloo", rank=rank, world_size=world_size)
    ddp = torch.distributed.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    return ddp

def ddp_cleanup(opt):
    if opt.amp :
        ADDP.destroy_process_group()
    else:
        DDP.destroy_process_group()
def main(opt):


    if torch.cuda.is_available():
        #torch.distributed.init_process_group(backend='gloo', init_method='env://')
        #torch.distributed.init_process_group(backend='nccl', init_method='env://')

        num_gpus = torch.distributed.get_world_size()
        torch.cuda.manual_seed(123)
        #num_gpus = 1
    else:
        torch.manual_seed(123)
        num_gpus = 1

    train_params = {"batch_size": opt.batch_size * num_gpus,
                    "shuffle": True,
                    "drop_last": False,
                    "num_workers": opt.num_workers,
                    "collate_fn": collate_fn}

    test_params = {"batch_size": opt.batch_size * num_gpus,
                   "shuffle": False,
                   "drop_last": False,
                   "num_workers": opt.num_workers,
                   "collate_fn": collate_fn}

    if opt.model == "ssd":
        dboxes = generate_dboxes(model="ssd")
        model = SSD(backbone=ResNet(), num_classes=len(customcoco_classes))
    else:
        dboxes = generate_dboxes(model="ssdlite")
        model = SSDLite(backbone=MobileNetV2(), num_classes=len(customcoco_classes))
    train_set = BCDataset(opt.data_path,  "train", SSDTransformer(dboxes, (300, 300), val=False))
    train_loader = DataLoader(train_set, **train_params)
    test_set = BCDataset(opt.data_path, "test", SSDTransformer(dboxes, (300, 300), val=True))
    test_loader = DataLoader(test_set, **test_params)

    encoder = Encoder(dboxes)

    opt.lr = opt.lr * num_gpus * (opt.batch_size / 32)
    criterion = Loss(dboxes)

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=opt.multistep, gamma=0.1)

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

        if opt.amp:
            from apex import amp
            from apex.parallel import DistributedDataParallel as DDP
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        else:
            from torch.nn.parallel import DistributedDataParallel as DDP
        # It is recommended to use DistributedDataParallel, instead of DataParallel
        # to do multi-GPU training, even if there is only a single node.
            model = DDP(model)


    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    checkpoint_path = os.path.join(opt.save_folder, "SSD.pth")

    writer = SummaryWriter(opt.log_path)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        first_epoch = checkpoint["epoch"] + 1
        model.module.load_state_dict(checkpoint["model_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        first_epoch = 0

    for epoch in range(first_epoch, opt.epochs):
        train(model, train_loader, epoch, writer, criterion, optimizer, scheduler, opt.amp)
        evaluate(model, test_loader, epoch, writer, encoder, opt.nms_threshold)

        checkpoint = {"epoch": epoch,
                      "model_state_dict": model.module.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)


def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

def sample_dataloader(opt, device):
    train_params = {"batch_size": opt.batch_size,
                    "shuffle": True,
                    "drop_last": False,
                    "num_workers": opt.num_workers,
                    "collate_fn": collate_fn}
    if opt.model == "ssd":
        dboxes = generate_dboxes(model="ssd")
    else:
        dboxes = generate_dboxes(model="ssdlite")
    train_set = BCDataset(opt.data_path,  "train", SSDTransformer(dboxes, (300, 300), val=False))

    # import fiftyone as fo
    # dataset = fo.Dataset(
    #     DATA_PATH,
    #     dataset_type=fo.types.COCODetectionDataset,
    #     name="customcoco"
    # )
    # session = fo.launch_app(train_set)
    #
    train_loader = DataLoader(train_set, **train_params)
    # DataLoader is iterable over Dataset
    for img, img_id, img_size, _, _ in train_loader:
        cv2.imshow(img)
        print(img_id)

if __name__ == "__main__":

    opt = get_args()
    device = check_cuda()
    #sample_dataloader(opt,device=device)
    ddp=ddp_setup(opt, 0, 1)
    main(opt)
    ddp.destroy_process_group()
