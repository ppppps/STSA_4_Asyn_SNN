import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from tqdm import tqdm

from datasets.dvs128 import DVS128Gesture
from functions import TET_loss, seed_all
from models.settings_dvs128 import args
from models.STSTransformer_dvs128 import STSTransformer

gpu_list = args.gpu
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_all(2022)


def train(model, device, train_loader, criterion, optimizer, args):
    running_loss = 0
    model.train()
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(train_loader):
        with torch.autograd.set_detect_anomaly(True):
            model.zero_grad()
            optimizer.zero_grad()
            labels = labels.to(device)
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.transpose(0, 1)
            mean_out = outputs.mean(0)
            if args.TET:
                loss = TET_loss(outputs, labels, criterion, args.means, args.lamb)
            else:
                loss = criterion(mean_out, labels)
            running_loss = running_loss + loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
            total += float(labels.size(0))
            _, predicted = mean_out.cpu().max(1)
            correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total


@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = outputs.transpose(0, 1)
        mean_out = outputs.mean(0)
        _, predicted = mean_out.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
    final_acc = 100 * correct / total
    return final_acc


def dvs_aug(data):
    off1 = random.randint(-5, 5)
    off2 = random.randint(-5, 5)
    data = np.roll(data, shift=(off1, off2), axis=(2, 3))
    return data


if __name__ == '__main__':
    patch_size = 16
    token_len = 256
    block_num = 2
    head_num = 1
    tau = 0.5

    root_dir = 'E:\\Yan\\spiking\\code\\dvs128'

    train_set = DVS128Gesture(root_dir, train=True, data_type='event')
    test_set = DVS128Gesture(root_dir, train=False, data_type='event')

    if args.data_aug:
        train_set = DVS128Gesture(root=root_dir, train=True, data_type='frame', frames_number=args.T, split_by='number',
                                  transform=dvs_aug, overlap=0.)
        test_set = DVS128Gesture(root=root_dir, train=False, data_type='frame', frames_number=args.T, split_by='number',
                                 transform=None, overlap=0.)

    else:
        train_set = DVS128Gesture(root=root_dir, train=True, data_type='frame', frames_number=args.T, split_by='number',
                                  transform=None, overlap=0.)
        test_set = DVS128Gesture(root=root_dir, train=False, data_type='frame', frames_number=args.T, split_by='number',
                                 transform=None, overlap=0.)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=False, drop_last=True)

    model = STSTransformer(
        proj_drop_rate=0.,
        attn_drop_rate=0.,
        forward_drop_rate=0.3,
        img_size_h=128, img_size_w=128,
        patch_size=patch_size, token_len=token_len, num_heads=head_num, mlp_ratios=1,
        in_channels=2, num_classes=11, depths=block_num,
        T=args.T,
        tau=tau
    )

    # model = nn.DataParallel(model)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-6, T_max=args.epochs)

    best_acc = 0
    best_epoch = 0
    train_acc = []
    test_acc = []
    for epoch in tqdm(range(args.epochs)):

        loss, acc = train(model, device, train_loader, criterion, optimizer, args)
        print('train: ', acc, loss)
        train_acc.append(acc)
        scheduler.step()
        facc = test(model, test_loader, device)
        print('test: ', facc)
        test_acc.append(facc)

        if best_acc < facc:
            best_acc = facc
            best_epoch = epoch + 1
            # torch.save(parallel_model.module.state_dict(), 'STSTrans_dvs128.pth')
        print('BestTestAcc: ', best_acc)
        print('\n')
    print(train_acc)
    print(test_acc)
