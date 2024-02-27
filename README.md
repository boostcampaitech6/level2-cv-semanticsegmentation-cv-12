# Hand Bone Image Segmentation

![image](https://github.com/boostcampaitech6/level2-cv-semanticsegmentation-cv-12/assets/56228633/29e66c66-a953-4563-a2d1-7c853fc6ee75)

- 2024.02.07 ~ 2024.02.22
- 네이버 커넥트 재단 및 Upstage에서 주관하는 비공개 대회
- X-ray 이미지에서 사람의 뼈를 Segmentation

## Members

> 공통: EDA
>
>[김세진](https://github.com/Revabo): Augmenatation 실험, Multi Loss 실험
>
>[박혜나](https://github.com/heynapark): Github 초기 세팅, base code 모듈화, 데이터 전처리 실험
>
>[이동우](https://github.com/Dong-Uri): Flip 실험, Ensemble, DeepLabV3 실험
>
>[진민주](https://github.com/freenozero): 전처리, Augmentation 실험, Unet++ 실험
>
>[허재영](https://github.com/jae-heo): mmsegmentation 세팅


## 문제 정의(대회소개) & Project Overview
Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나입니다. 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있습니다. 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적입니다. Bone Segmentation은 다음과 같은 상황에서 활용됩니다.

1. 질병, 골절, 변형 등의 파악
2. 수술 계획을 위한 뼈 구조 분석
3. 의료장비 제작에서의 정보 제공
4. 의료 교육에서의 활용


## 대회 결과

Public, Private 2등!
![image](https://github.com/boostcampaitech6/level2-cv-semanticsegmentation-cv-12/assets/56228633/54d75db4-3991-4448-8aa9-78f86e387c10)
![image](https://github.com/boostcampaitech6/level2-cv-semanticsegmentation-cv-12/assets/56228633/27a8511e-5678-45f6-a7ca-5c3fdedaac4f)

## Dataset

1) Dataset

- Train : 800 Labeled Images
- Test : 288 Unlabeled Images
- 2048 x 2048, 3channel
- 개인별 왼손, 오른손 두 장의 이미지 set

2) Label
- 29 Classes
```
'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
'Triquetrum', 'Pisiform', 'Radius', 'Ulna'
```

## Metric
- Test set의 Dice coefficient로 평가
    - Semantic Segmentation에서 사용되는 대표적인 성능 측정 방법
    - Dice
        $$Dice(A,B) = \frac{2 * |A \cap B|}{|A| + |B|}$$

## Tools

- Github
- Notion
- Slack
- Wandb

## Project Outline
![image](https://github.com/boostcampaitech6/level2-cv-semanticsegmentation-cv-12/assets/56228633/09925912-7a28-48d2-a7c8-626678364040)


## Models & Backbones

- FCN: ResNet50
- DeepLabV3+: Xception
- Unet++: Resnet34, ResNet101, efficientNet_d4

## Preprocessing

- Resize
- Sharpen
- CLAHE

## Data Augmentations

- HorizontalFlip
- Rotate
- Crop

### Combine Loss
- DICE + BCE
- DICE + IOU
