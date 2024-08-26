import os
import sys
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import jittor as jt
from jittor import nn
from jittor import transform
from jittor.dataset import Dataset
from jittor.lr_scheduler import CosineAnnealingLR
from jclip.convnext import convnext_base,convnext_large
from jclip.convnextv2 import convnextv2_base
from jittor.optim import SGD
from jclip.transnext import transnext_base
from scipy.ndimage.filters import gaussian_filter
import random
from random import choice, shuffle

# from jclip.mmla import mmla_base
# from jclip.RMT import RMT_M2
# from jittor.models import Resnet50

jt.flags.use_cuda = 1


    
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    """1.论文里的公式2，求出B的rw,rh"""
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    """2.论文里的公式2，求出B的rx,ry（bbox的中心点）"""
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    # 限制坐标区域不超过样本大小

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    """3.返回剪裁B区域的坐标值"""
    return bbx1, bby1, bbx2, bby2




def get_train_transforms():
    return transform.Compose([
        # transform.Lambda(lambda img: data_augment(img)),
        transform.Resize((320, 320)),
        transform.RandomCrop((280,280)),
        
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.ImageNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


def get_valid_transforms():
    return transform.Compose([
        transform.Resize(384),
        transform.ToTensor(),
        transform.ImageNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

imgs_dir = 'Dataset/'


cutmix = 0
class CUB200(Dataset):
    def __init__(self, img_path, img_label, batch_size, part='train', shuffle=False, transform=None):
        super(CUB200, self).__init__()
        self.img_path = img_path
        self.img_label = img_label
        self.transform = transform
        self.set_attrs(
            batch_size=batch_size,
            total_len=len(self.img_path),
            shuffle=shuffle
        )


    def __getitem__(self, index):
        img = os.path.join(imgs_dir, self.img_path[index])
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            
        img = np.asarray(img)
            
        label = self.img_label[index]
        
        return img, label


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, accum_iter, scheduler):
    model.train()
    total_acc = 0
    total_num = 0
    losses = []
    


    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]')
    for i, (train_img, labels) in enumerate(pbar):
        
        if cutmix:
            # generate mixed sample
            """1.设定lamda的值，服从beta分布"""
            lam = np.random.beta(1, 1)
            """2.找到两个随机样本"""
            rand_index = jt.randperm(train_img.size()[0])
            target_a = labels  # 一个batch
            target_b = labels[rand_index]  # 将原有batch打乱顺序
            """3.生成剪裁区域B"""
            bbx1, bby1, bbx2, bby2 = rand_bbox(train_img.size(), lam)
            """4.将原有的样本A中的B区域，替换成样本B中的B区域"""
            # 打乱顺序后的batch组和原有的batch组进行替换[对应id下]
            train_img[:, :, bbx1:bbx2, bby1:bby2] = train_img[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            """5.根据剪裁区域坐标框的值调整lam的值"""
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (train_img.size()[-1] * train_img.size()[-2]))
            # compute output
            """6.将生成的新的训练样本丢到模型中进行训练"""
            output = model(train_img)
            """7.按lamda值分配权重"""
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            output = model(train_img)
            # print(output.shape)

            loss = criterion(output, labels)            

        # print(images.shape)

        
        # labels = Variable(labels.view(-1)).cuda()


        
        optimizer.step(loss)

        # optimizer.backward(loss)
        # if (i + 1) % accum_iter == 0 or i + 1 == len(train_loader):
        # optimizer.step(loss)
        
        # acc, train_acc3 = accuracy(pred, labels, topk=(1, 3))

        # print(output)
        pred = np.argmax(output.numpy(), axis=1)
        acc = np.sum(pred == labels.numpy())
        total_acc += acc
        total_num += labels.shape[0]
        losses.append(loss.data[0])

        pbar.set_description(f'Epoch {epoch} loss={sum(losses) / len(losses):.2f}'
                             f'acc={total_acc / total_num:.2f}')
    scheduler.step()


def valid_one_epoch(model, val_loader, epoch):
    model.eval()
    total_acc = 0
    total_num = 0

    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [VALID]')
    for images, labels in val_loader:
        # print(labels)
        

        output = model(images)
        
        
        # acc, val_acc3 = accuracy(output, val_labels, topk=(1, 3))
        
        pred = np.argmax(output.numpy(), axis=1)
        # print(pred)

        acc = np.sum(pred == labels.numpy())
        
        # print(pred)
        # print(labels)
        total_acc += acc
        total_num += labels.shape[0]
        
        

        pbar.set_description(f'Epoch {epoch}' f'acc={total_acc / total_num:.4f}')

    acc = total_acc / total_num
    return acc


if __name__ == '__main__':
    jt.set_global_seed(2024)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--eta_min', type=float, default=1e-5)
    parser.add_argument('--T_max', type=int, default=15)
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--accum_iter', type=int, default=1)
    args = parser.parse_args()
    
    options = {
        'num_classes': 130,
        'threshold': 0.74,
        'lr': args.lr,
        'eta_min': args.eta_min,
        'T_max': args.T_max,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'accum_iter': args.accum_iter,
    }
    
    train_data = open('train_data/train_711.txt').read().splitlines()
    val_data = open('train_data/val_711.txt').read().splitlines()

    train_imgs,train_labels=[],[]
    val_imgs,val_labels=[],[]
    num = 0
    for l in train_data:
        a = int(l.split(',')[1])
        b = l.split(',')[0]
        if a >= 244:
            train_imgs.append(b)
            train_labels.append(a-244)


    for l in val_data:
        num = num+1
        a = int(l.split(',')[1])
        b = l.split(',')[0]
        if a >= 244 and num % 2 == 0:
            val_imgs.append(b)
            val_labels.append(a-244)
        
    train_loader = CUB200(train_imgs, train_labels, 32, 'train', shuffle=True,
                          transform=get_train_transforms())
    val_loader = CUB200(val_imgs, val_labels, 32, 'valid', shuffle=True,
                        transform=get_valid_transforms())



    
    model = convnextv2_base(num_classes=130)
    state_dict = jt.load('pretrain/convnextv2_base_1k_224_ema.pkl')
    # model = convnextv2_base(num_classes=130)


    model.load_parameters(state_dict)

    criterion = nn.CrossEntropyLoss()
    # optimizer = SGD(model.head.parameters(),lr=options['lr'],momentum=0.9)
    
    optimizer = nn.AdamW(model.head.parameters(), options['lr'])
    scheduler = CosineAnnealingLR(optimizer, options['T_max'], options['eta_min'])

    best_acc = options['threshold']
    for epoch in range(options['epochs']):
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, options['accum_iter'], scheduler)
        if epoch >= 30:
        
            acc = valid_one_epoch(model, val_loader, epoch)
            print(acc)
            if acc > best_acc:
                best_acc = acc
                model.save(f'out/connext2_base-{epoch}-{acc:.3f}.pkl')
