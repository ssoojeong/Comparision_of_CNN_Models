# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import logging
from tqdm import tqdm
from datetime import datetime

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', type=str, default=None)
parser.add_argument('--save_log', type=str, default='./results/log')
parser.add_argument('--save_model', type=str, default='./results/model')
parser.add_argument('--gpus', default='0', type=int, help='gpu_number')
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--epochs', default=25, type=int)
parser.add_argument('--name', default='tutorial', type=str)
parser.add_argument('--model', default='resnet18', type=str)
args = parser.parse_args()


#gpu 할당
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.gpus}"

cudnn.benchmark = True
plt.ion()   # 대화형 모드

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
# 로그 생성
logger = logging.getLogger()
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
mkdir(args.save_log)
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
args.name = current_time

if args.pretrained:
    file_handler = logging.FileHandler(os.path.join(args.save_log, f'{args.name}_{args.model}_{args.epochs}epoch_pre_train.log'))
else:
    file_handler = logging.FileHandler(os.path.join(args.save_log, f'{args.name}_{args.model}_{args.epochs}epoch_train.log'))
logger.addHandler(file_handler)


def imshow(inp, title=None):
    """tensor를 입력받아 일반적인 이미지로 보여줍니다."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 갱신이 될 때까지 잠시 기다립니다.



# 학습을 위해 데이터 증가(augmentation) 및 일반화(normalization)
# 검증을 위한 일반화
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'pred: {class_names[preds[j]]}\nreal: {class_names[labels[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    logger.info("############학습 시작############")
    logger.info('-' * 10)
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch}/{num_epochs - 1}')

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터 반복 및 진행률 표시
            phase_dataloader = dataloaders[phase]
            phase_size = len(phase_dataloader)
            
            with tqdm(total=phase_size, desc=f"{phase.capitalize()} Progress", leave=False) as pbar:
                # 데이터를 반복
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 매개변수 경사도를 0으로 설정
                    optimizer.zero_grad()

                    # 순전파
                    # 학습 시에만 연산 기록을 추적
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # 학습 단계인 경우 역전파 + 최적화
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # 통계
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # 진행률 업데이트
                    pbar.update(1)
                    pbar.set_postfix(Loss=loss.item())

                    
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}') #로그 기록용으로 변경

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        logger.info('-' * 10)
        print()

    time_elapsed = time.time() - since
    logger.info("############학습 끝############")
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best val Acc: {best_acc:4f}')

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
    return model


def SaveModel(model, path):
    mkdir(path)
    if args.pretrained:
        torch.save(model.state_dict(), os.path.join(path, f'{args.name}_{args.model}_{args.epochs}epoch_pre_model_.bin'))
    else:
        torch.save(model.state_dict(), os.path.join(path, f'{args.name}_{args.model}_{args.epochs}epoch_model.bin'))


if __name__ == '__main__':

    #데이터 로드
    data_dir = args.data
    if data_dir is None:
        # cifar10 데이터셋 로드
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms['val'])

        image_datasets = {'train': train_dataset, 'val': val_dataset}
    else: 
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    #gpu 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    if args.model == 'resnet18':
        model = torchvision.models.resnet18(weights="DEFAULT")
    if args.model == 'resnet50':
        model = torchvision.models.resnet50(weights="DEFAULT")
    if args.model == 'vgg16':
        model = torchvision.models.vgg16(weights="DEFAULT") 
    if args.model == 'alexnet':
        model = torchvision.models.alexnet(weights="DEFAULT") 
    if args.model == 'inception_v3':
        model = torchvision.models.inception_v3(weights="DEFAULT") 
    if args.model == 'googlenet':
        model = torchvision.models.googlenet(weights="DEFAULT") 
    if args.model == 'mobilenet_v2':
        model = torchvision.models.mobilenet_v2(weights="DEFAULT")
    if args.model == 'densenet121':
        model = torchvision.models.densenet121(weights="DEFAULT") 

    if args.pretrained:
        for param in model.parameters():
            param.requires_grad = False

    # 새로 생성된 모듈의 매개변수는 기본값이 requires_grad=True 임
    try:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(image_datasets['train'].classes))
    except:
        if args.model == 'densenet121':
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, len(image_datasets['train'].classes))
        else:
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, len(image_datasets['train'].classes))
        
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # 이전과는 다르게 마지막 계층의 매개변수들만 최적화되는지 관찰
    try:    
        optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    except:
        if args.model == 'densenet121':
            optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
        else:
            optimizer = optim.SGD(model.classifier[-1].parameters(), lr=0.001, momentum=0.9)
    # 7 에폭마다 0.1씩 학습률 감소
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    
    model = train_model(model, criterion, optimizer,
                         exp_lr_scheduler, num_epochs=args.epochs)
    mkdir(args.save_model)
    SaveModel(model, args.save_model)