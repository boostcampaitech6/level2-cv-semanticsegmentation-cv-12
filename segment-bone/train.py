# python native
import os
import random
import datetime

# external library
import cv2
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.optim.lr_scheduler as lr_scheduler

from dataset import XRayDataset
from loss import calc_loss
from model import DeepLabV3p, FCNResNet50, UnetPlusPlus # 모델 import

import wandb

from config import WANDB_GROUP, WANDB_NAME, KFOLD_N, PRETRAINED, PRETRAINED_DIR, IMAGE_ROOT, LABEL_ROOT, BATCH_SIZE, BATCH_SIZE_VALID, LR, RANDOM_SEED, NUM_EPOCHS, VAL_EVERY, SAVED_DIR, CLASSES, WANDB_NAME


wandb.init(
    # set the wandb project where this run will be logged
    project="segmentation",
    entity='cv-12',
    group=WANDB_GROUP,
    name=WANDB_NAME,
)

if not os.path.isdir(SAVED_DIR):
    os.mkdir(SAVED_DIR)

# 파일 찾기
pngs = {
    os.path.relpath(os.path.join(root, fname), start=IMAGE_ROOT)
    for root, _dirs, files in os.walk(IMAGE_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".png"
}

print(f'Find {len(pngs)} pngs')

jsons = {
    os.path.relpath(os.path.join(root, fname), start=LABEL_ROOT)
    for root, _dirs, files in os.walk(LABEL_ROOT)
    for fname in files
    if os.path.splitext(fname)[1].lower() == ".json"
}

print(f'Find {len(jsons)} jsons')

# 페어 확인 및 이름 순 정렬
jsons_fn_prefix = {os.path.splitext(fname)[0] for fname in jsons}
pngs_fn_prefix = {os.path.splitext(fname)[0] for fname in pngs}

assert len(jsons_fn_prefix - pngs_fn_prefix) == 0
assert len(pngs_fn_prefix - jsons_fn_prefix) == 0

pngs = sorted(pngs)
jsons = sorted(jsons)

pngs = np.array(pngs)
jsons = np.array(jsons)

# split train-valid
# 한 폴더 안에 한 인물의 양손에 대한 `.png` 파일이 존재하기 때문에
# 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
# 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
groups = [os.path.dirname(fname) for fname in pngs]

# dummy label
ys = [0 for fname in pngs]

# 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
# 5으로 설정하여 GroupKFold를 수행합니다.
gkf = GroupKFold(n_splits=5)

train_filenames = []
train_labelnames = []
valid_filenames = []
valid_labelnames = []
for i, (x, y) in enumerate(gkf.split(pngs, ys, groups)):
    # 0번을 validation dataset으로 사용합니다.
    if i == KFOLD_N:
        valid_filenames += list(pngs[y])
        valid_labelnames += list(jsons[y])

    else:
        train_filenames += list(pngs[y])
        train_labelnames += list(jsons[y])

tf = A.Resize(512, 512)

train_dataset = XRayDataset(train_filenames, train_labelnames, transforms=tf, is_train=True)
valid_dataset = XRayDataset(valid_filenames, valid_labelnames, transforms=tf, is_train=False)

image, label = train_dataset[0]

print(f'Images and Labels size : {image.shape}, {label.shape}')

print(f'Train dataset : {len(train_dataset)}')
print(f'Valid dataset : {len(valid_dataset)}')

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    drop_last=True,
)

valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=BATCH_SIZE_VALID,
    shuffle=False,
    num_workers=2, # 너무 커지면 메모리 에러 날 수 있다고 함
    drop_last=False
)

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)

    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def save_model(model, file_name='best_model.pt'):
    output_path = os.path.join(SAVED_DIR, file_name)
    torch.save(model, output_path)

def set_seed():
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.cuda()
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()

            outputs = model(images)

            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)

            # restore original size
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1

            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()

            dice = dice_coef(outputs, masks)
            dices.append(dice)

    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)

    avg_dice = torch.mean(dices_per_class).item()

    wandb.log({'Epoch': epoch, 'avg_dice': avg_dice})

    return avg_dice

def train(model, data_loader, val_loader, criterion, optimizer, scheduler):
    print(f'Start training..')
    model.cuda()

    n_class = len(CLASSES)
    best_dice = 0.

    for epoch in range(NUM_EPOCHS):
        model.train()

        for step, (images, masks) in enumerate(data_loader):
            # gpu 연산을 위해 device 할당
            images, masks = images.cuda(), masks.cuda()


            # inference
            outputs = model(images)

            # loss 계산
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # step 주기에 따른 loss 출력
            if (step + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}, '
                    f'LR: {current_lr}'
                )

                wandb.log({'Epoch': epoch + 1, 'loss': loss.item(), 'LR': current_lr})

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % VAL_EVERY == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)

            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {SAVED_DIR}")
                best_dice = dice
                save_model(model)

        scheduler.step()

model = DeepLabV3p(in_channels=3, num_classes=len(CLASSES))

if PRETRAINED:
    model = torch.load(PRETRAINED_DIR)

# Loss function 정의
# criterion = nn.BCEWithLogitsLoss()
criterion = calc_loss

# Optimizer 정의
optimizer = optim.Adam(params=model.parameters(), lr=LR, weight_decay=1e-6)

# Scheduler 정의
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

set_seed()

train(model, train_loader, valid_loader, criterion, optimizer, scheduler)