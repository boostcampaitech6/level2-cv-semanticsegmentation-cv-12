# inference.py

import os
import numpy as np
import cv2
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import pandas as pd
import torch.nn.functional as F
import albumentations as A
import matplotlib.pyplot as plt

from dataset import XRayInferenceDataset
from config import CLASSES, IND2CLASS, PALETTE, TEST_IMAGE_ROOT


# mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.

def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


# RLE로 인코딩된 결과를 mask map으로 복원합니다.

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)

def label2rgb(label):
    image_size = label.shape[1:] + (3, )
    image = np.zeros(image_size, dtype=np.uint8)
    
    for i, class_label in enumerate(label):
        image[class_label == 1] = PALETTE[i]
        
    return image

def test(model, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    rles = []
    filename_and_class = []
    with torch.no_grad():
        n_class = len(CLASSES)

        for step, (images, image_names) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = images.cuda()    
            outputs = model(images)['out']
            
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()
            
            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{IND2CLASS[c]}_{image_name}")
                    
    return rles, filename_and_class




if __name__ == "__main__":
    # 모델 설정
    model = torch.load("/data/ephemeral/home/saved_model/best_model_epoch50.pt")

    # 테스트 데이터셋 및 데이터로더 설정
    tf = A.Resize(512, 512)
    test_dataset = XRayInferenceDataset(TEST_IMAGE_ROOT, transforms=tf)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=16,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )
    # Inference 실행
    rles, filename_and_class = test(model, test_loader)
    

    # 결과 저장
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })
    df.head(10)

    df.to_csv("output.csv", index=False)


    image_path = os.path.join(TEST_IMAGE_ROOT, filename_and_class[0].split("_")[1])
    print(image_path)
    image = cv2.imread(image_path)

    preds = []
    for rle in rles[:len(CLASSES)]:
        pred = decode_rle_to_mask(rle, height=2048, width=2048)
        preds.append(pred)

    preds = np.stack(preds, 0)

    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].imshow(image)    # remove channel dimension
    ax[1].imshow(label2rgb(preds))

    # plt.show()
    plt.savefig("output_vis.png")