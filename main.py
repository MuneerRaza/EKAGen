import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
import argparse
import os
from models import utils, caption
from datasets import xray
from utils.engine import train_one_epoch, evaluate
from models.model import swin_tiny_patch4_window7_224 as create_model
from utils.stloss import SoftTarget
import re
import warnings
from config import parse_args

warnings.filterwarnings("ignore", category=FutureWarning, message="Support for mismatched key_padding_mask and attn_mask is deprecated")



def build_diagnosisbot(num_classes, detector_weight_path):
    model = create_model(num_classes=num_classes)
    assert os.path.exists(detector_weight_path), "file: '{}' dose not exist.".format(detector_weight_path)
    model.load_state_dict(torch.load(detector_weight_path, map_location=torch.device('cpu'), weights_only=False), strict=True )
    for k, v in model.named_parameters():
        v.requires_grad = False
    return model


def build_tmodel(config, device):
    tmodel, _ = caption.build_model(config)
    print("Loading teacher medel Checkpoint...")
    tcheckpoint = torch.load(config.t_model_weight_path, map_location='cpu', weights_only=False)
    tmodel.load_state_dict(tcheckpoint['model'])
    tmodel.to(device)
    return tmodel


def main(config):
    config = parse_args()
    print(config)
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')

    if os.path.exists(config.thresholds_path):
        with open(config.thresholds_path, "rb") as f:
            thresholds = pickle.load(f)

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    detector = build_diagnosisbot(config.num_classes, config.detector_weight_path)
    detector.to(device)

    model, criterion = caption.build_model(config)
    criterionKD = SoftTarget(4.0)
    model.to(device)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": config.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=config.lr, weight_decay=config.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.lr_drop)

    dataset_train = xray.build_dataset(config, mode='training', anno_path=config.anno_path, data_dir=config.data_dir,
                                       dataset_name=config.dataset_name, image_size=config.image_size,
                                       theta=config.theta, gamma=config.gamma, beta=config.beta)
    dataset_val = xray.build_dataset(config, mode='validation', anno_path=config.anno_path, data_dir=config.data_dir,
                                     dataset_name=config.dataset_name, image_size=config.image_size,
                                     theta=config.theta, gamma=config.gamma, beta=config.beta)
    dataset_test = xray.build_dataset(config, mode='test', anno_path=config.anno_path, data_dir=config.data_dir,
                                      dataset_name=config.dataset_name, image_size=config.image_size,
                                      theta=config.theta, gamma=config.gamma, beta=config.beta)
    print(f"Train: {len(dataset_train)}")
    print(f"Valid: {len(dataset_val)}")
    print(f"Test: {len(dataset_test)}")

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=config.num_workers,
        collate_fn=dataset_train.collate_fn)
    data_loader_val = DataLoader(dataset_val, config.batch_size,
                                 sampler=sampler_val, drop_last=False,
                                 collate_fn=dataset_val.collate_fn)

    data_loader_test = DataLoader(dataset_test, config.batch_size,
                                  sampler=sampler_test, drop_last=False,
                                  collate_fn=dataset_test.collate_fn)
    if config.mode == "train":
        tmodel = build_tmodel(config, device)
        

        if config.resume_state:
            weights_dict = torch.load(config.resume_state, map_location='cpu', weights_only=True)['model']
            model.load_state_dict(weights_dict, strict=False)
            print("Resuming Training...")
            match = re.search(r"epoch(\d+)", config.resume_state)
            epoch_number = int(match.group(1))
            config.start_epoch = epoch_number

        else:
            print("Start Training...")
        for epoch in range(config.start_epoch, config.epochs):
            print(f"Epoch: {epoch}")
            x_epoch = epoch+1
            epoch_loss = train_one_epoch(
                model, tmodel, detector, criterion, criterionKD, data_loader_train, optimizer, device,
                config.clip_max_norm, thresholds=thresholds, tokenizer=dataset_train.tokenizer, config=config)
            lr_scheduler.step()
            print(f"Training Loss: {epoch_loss}")

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, config.model_dir + "/" + config.dataset_name + "_weight_epoch" + str(epoch) + "_.pth")
            
            if x_epoch % config.validation_freq == 0:
                validate_result = evaluate(model, detector, criterion, data_loader_val, device, config,
                                        thresholds=thresholds, tokenizer=dataset_val.tokenizer)
                print(f"validate_result: {validate_result}")
            if x_epoch % config.test_freq == 0:
                test_result = evaluate(model, detector, criterion, data_loader_test, device, config,
                                    thresholds=thresholds, tokenizer=dataset_test.tokenizer)
                print(f"test_result: {test_result}")
    if config.mode == "test":
        if os.path.exists(config.model_path):
            weights_dict = torch.load(config.model_path, map_location='cpu', weights_only=False)['model']
            model.load_state_dict(weights_dict, strict=False)
            model.to(device)

        print("Start Testing..")
        test_result = evaluate(model, detector, criterion, data_loader_test, device, config,
                               thresholds=thresholds, tokenizer=dataset_test.tokenizer)
        print(f"test_result: {test_result}")


if __name__ == "__main__":
    main()
