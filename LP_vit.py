import jittor as jt
from PIL import Image
import jclip as clip
import os
from tqdm import tqdm
import argparse
from sklearn.linear_model import LogisticRegression
import numpy as np
from jittor import transform
from sklearn.metrics import accuracy_score
import random
from sklearn import svm
import jittor.nn as nn
from jittor.transform import CenterCrop, ImageNormalize, Compose, _setup_size, to_pil_image, resize,RandomHorizontalFlip
from jclip.clip import _convert_image_to_rgb,ImageToTensor,Resize
from jittor.dataset import Dataset
from scipy.linalg import eigh
import pickle


jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='A')

args = parser.parse_args()

clip_model, preprocess = clip.load("pretrain/ViT-B-32.pkl")

# with open('class_describe.pickle', 'rb') as file:   #用with的优点是可以不用写关闭文件操作
#     dict_get = pickle.load(file)
# print(dict_get)


classes = open('Dataset/classes.txt').read().splitlines()


# template = ['a photo of a {}.']

template = []

new_classes = []
for c in classes:
    c = c.split(' ')[0]
    if c.startswith('Animal'):
        c = c[7:]
        a = 'a photo of a ' + c 
    if c.startswith('Thu-dog'):
        c = c[8:]
        a = 'a photo of a ' + c + ', a type of dog'
        # c = 'dog'
    if c.startswith('Caltech-101'):
        c = c[12:]
        a = 'a photo of a ' + c 
    if c.startswith('Food-101'):
        c = c[9:]
        a = 'a photo of a ' + c + ', a type of food '

    new_classes.append(c)
    template.append(a)

new_classes = list(new_classes)
print('classes name:',new_classes)
print('clasees length:',len(new_classes))
# print(new_classes)

def clip_classifier(classnames, template, clip_model):
    with jt.no_grad():
        clip_weights = []
        num = 0
        for classname in classnames:
            # Tokenize the prompts
            # classname = classname.replace('_', ' ')
            texts = template[num].replace('_', ' ')
            texts = clip.tokenize(texts)
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
            num += 1

        clip_weights = jt.stack(clip_weights, dim=1)
    return clip_weights

clip_weights = clip_classifier(
    new_classes, template, clip_model)


def get_train_transforms(n_px):
    return Compose([
        Resize(224, mode=Image.BICUBIC),
        CenterCrop(n_px),
        RandomHorizontalFlip(0.5),
        _convert_image_to_rgb,
        ImageNormalize((0.48145466, 0.4578275, 0.40821073),
                       (0.26862954, 0.26130258, 0.27577711)),
        ImageToTensor()
    ])

def get_valid_transforms(n_px):
    return Compose([
        Resize(224, mode=Image.BICUBIC),
        CenterCrop(n_px),
        RandomHorizontalFlip(0.5),
        _convert_image_to_rgb,
        ImageNormalize((0.48145466, 0.4578275, 0.40821073),
                       (0.26862954, 0.26130258, 0.27577711)),
        ImageToTensor()
    ])

def get_test_transforms(n_px):
    return Compose([
        Resize(224, mode=Image.BICUBIC),
        CenterCrop(n_px),
        RandomHorizontalFlip(0.5),
        _convert_image_to_rgb,
        ImageNormalize((0.48145466, 0.4578275, 0.40821073),
                       (0.26862954, 0.26130258, 0.27577711)),
        ImageToTensor()
    ])

imgs_dir = 'Dataset/'

def pre_load_features(clip_model, loader):
    features, labels = [], []
    with jt.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)

    features, labels = jt.cat(features), jt.cat(labels)
    
    return features, labels


print("Getting textual features as CLIP's classifier.")
clip_weights = clip_classifier(
    new_classes, template, clip_model)


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
    
class CUB200_test(Dataset):
    def __init__(self, img_path, img_label, batch_size, part='train', shuffle=False, transform=None):
        super(CUB200_test, self).__init__()
        self.img_path = img_path
        self.img_label = img_label
        self.transform = transform
        self.set_attrs(
            batch_size=batch_size,
            total_len=len(self.img_path),
            shuffle=shuffle
        )


    def __getitem__(self, index):
        img = os.path.join(test_imgs_dir, self.img_path[index])
        img = Image.open(img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            
        img = np.asarray(img)
            
        label = self.img_label[index]
        
        return img, self.img_path[index]

print("Start load train set , val set, test set.\n")
print('Train Set (train.txt) make Sure 4 shot form code(process.py)')

imgs_dir = 'Dataset/'
train_data = open('train_data/train_711.txt').read().splitlines()
val_data = open('train_data/val_711.txt').read().splitlines()

test_imgs_dir = 'Dataset/TestSet' + args.split
test_imgs = os.listdir(test_imgs_dir)

label = len(test_imgs) * [0]


train_imgs,train_labels=[],[]
val_imgs,val_labels=[],[]
num = 0
for l in train_data:
    a = int(l.split(',')[1])
    b = l.split(',')[0]
    train_imgs.append(b)
    train_labels.append(a)
    
for l in val_data:
    num = num+1
    a = int(l.split(',')[1])
    b = l.split(',')[0]
    if num % 50 == 0:
        val_imgs.append(b)
        val_labels.append(a)
        

train_loader = CUB200(train_imgs, train_labels, 64, 'train', shuffle=True,
                      transform=get_train_transforms(224))
val_loader = CUB200(val_imgs, val_labels, 64, 'valid', shuffle=True,
                    transform=get_valid_transforms(224))

test_loader = CUB200_test(test_imgs, label, 128, 'test', shuffle=False,
                      transform=get_test_transforms(224))


print("\nExtracting visual features and labels from train set.")
features, labels = [], []
with jt.no_grad():
    for i, (images, target) in enumerate(tqdm(train_loader)):
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        features.append(image_features)
        labels.append(target)  
features, labels = jt.cat(features), jt.cat(labels)
print(features.shape)


print("\nExtracting visual features and labels from val set.")
val_features, val_labels = [], []
with jt.no_grad():
    for i, (images, target) in enumerate(tqdm(val_loader)):
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        val_features.append(image_features)
        val_labels.append(target)
val_features, val_labels = jt.cat(val_features), jt.cat(val_labels)

print("\nExtracting visual features and labels from teat A set.")

test_features, test_img = [], []
with jt.no_grad():
    for i, (images, target) in enumerate(tqdm(test_loader)):
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        test_features.append(image_features)
        test_img.append(target)
        
test_features = jt.cat(test_features)
test_img = np.concatenate(test_img)
print(test_img)

def get_one_hot(y_s, num_classes):
    """
        args:
            y_s : torch.Tensor of shape [n_task, shot]
        returns
            y_s : torch.Tensor of shape [n_task, shot, num_classes]
    """
    one_hot_size = list(y_s.size()) + [num_classes]
    
    one_hot = jt.zeros(one_hot_size)
    
    y_s = y_s.unsqueeze(-1)
    
    one_hot = jt.scatter(one_hot,-1, y_s, jt.array(1))

    return one_hot

def calculate_lr_alpha(features, clip_weights):
    # lr_alpha
    ftT = features @ clip_weights
    temp = jt.sum(jt.pow(ftT, 2),dim = 0)
    max_sum = max(temp)
    lr_alpha = features.shape[0] / (max_sum * 4)
    return lr_alpha

def compute_centroids_alpha(z_s,y_s):

    one_hot = get_one_hot(y_s, num_classes=y_s.unique().size(0))
    centroids = (one_hot*z_s/ one_hot.sum(-2, keepdim=True)).sum(1)  # [batch, K, d]
    return centroids


def compute_centroids(z_s,y_s):
    one_hot = get_one_hot(y_s, num_classes=y_s.unique().size(0)).transpose(1, 2)
    print(one_hot.shape)
    print(z_s.shape)
    # centroids = one_hot.bmm(z_s) / one_hot.sum(-1, keepdim=True)  # [batch, K, d]
    centroids = jt.bmm(one_hot,z_s)  # [batch, K, d]
    return centroids


# print(features.shape,labels.shape)
centroids = compute_centroids(features.unsqueeze(0), jt.array(labels.unsqueeze(0).numpy(),dtype=jt.int32))  # [batch, num_class, d]
# print(centroids[0].shape)


shot = 4

classifier = nn.Linear(features.shape[1],len(new_classes))


classifier.weight.data = centroids[0]

print(classifier.weight)


def calculate_init_alpha(features, labels, shots, clip_weights):
    # init_alpha
    alpha_tilde = compute_centroids_alpha((features @ clip_weights).unsqueeze(0), labels.unsqueeze(0))[0]
    alpha_tilde = alpha_tilde.double() * shots
    alpha_init = 250 / shots * alpha_tilde
    final_init_alpha_mean = jt.mean(alpha_init)
    return final_init_alpha_mean

def calculate_lr_w(features):
    # lr_w
    ff_t = features.transpose(0, 1) @ features
    ff_t_np = ff_t.numpy()
    w, v = eigh(ff_t_np)
    max_eigen = max(w) # check the iters of power iteration
    lr_w =  (4 * features.shape[0]) / max_eigen
    return lr_w

print('Running LP++')
# lr_w
lr_temp = calculate_lr_w(features)

final_init_alpha_mean= calculate_init_alpha(features, labels, 4, clip_weights)

final_init_alpha_mean = final_init_alpha_mean.astype(np.float32)

alpha_vec = jt.array(final_init_alpha_mean * jt.ones(1, len(new_classes)))

alpha_vec.astype(jt.float32)

import pickle

with open('out/vit_alpha_vec_711.pkl', 'wb') as file:
    pickle.dump(alpha_vec, file)

alpha_vec.requires_grad = True

print(alpha_vec.data)

# alpha_vec.requires_grad = True

# print(alpha_vec.data)

# lr_alpha
lr_alpha = calculate_lr_alpha(features, clip_weights)

print('Calculated lr_temp, lr_alpha:'.format(lr_temp, lr_alpha))

print('final_init_alpha_mean: {}'.format(final_init_alpha_mean))

optimizer = nn.SGD(classifier.parameters(), lr_temp, momentum=0.9)

from jittor.lr_scheduler import CosineAnnealingLR
        
# optimizer = nn.Adam(clip_ad_model.adapter.parameters(), 1e-3)
scheduler = CosineAnnealingLR(optimizer, 1000)

# optimizer = jt.optim.SGD(classifier.parameters(), lr_temp, momentum=0.9)


# Train
print('\nStart Training procedure!')

best_acc, best_epoch = 0.0, 0

criterion = nn.CrossEntropyLoss()

for epoch in range(500):

    print('Running model for epoch: {}'.format(epoch))
    classifier.train()

    vision_logits = classifier(features)
    text_logits = features @ clip_weights

    
    logits = vision_logits +  jt.ones(features.shape[0], 1) @ alpha_vec * text_logits
    
    
    loss = criterion(logits, labels)
    

    # if (epoch + 1) % 50 == 0:
    #     alpha_vec.data -= lr_alpha * jt.grad(logits, alpha_vec)
    
    pred = np.argmax(logits.data, axis=1)
    
    acc = np.mean(pred == labels.data)
    
    optimizer.step(loss)
    scheduler.step()

    
    classifier.eval()
    
    
    vision_logits_val = classifier(val_features)
    text_logits_val = val_features.detach() @ clip_weights
    logits_val = vision_logits_val + jt.ones(val_features.shape[0], 1) @ alpha_vec * text_logits_val

    pred = np.argmax(logits_val, axis=1)
    
    acc_val = np.mean(pred == val_labels.data)
    
    print('The accuracy for val data is ',acc_val)
    
    if acc_val > best_acc:
        best_acc = acc_val
        if best_acc > 0.73:
            classifier.save(f'out/LP-711-{best_acc:.3f}.pkl')
    
    
state_dict = jt.load(f'out/LP-711-{best_acc:.3f}.pkl')
classifier.load_parameters(state_dict)    

vision_logits_test = classifier(test_features)
text_logits_test = test_features.detach() @ clip_weights
logits_test = vision_logits_test + jt.ones(test_features.shape[0], 1) @ alpha_vec * text_logits_test
logits_test = logits_test.numpy()

# pred = np.argmax(logits_test, axis=1)
num = 0
with open('result.txt', 'w') as save_file:
    for i in test_img:
        top5_idx = logits_test[num].argsort()[-1:-6:-1]
        top5_idx = list(top5_idx)
        save_file.write(i + ' ' +
                        ' '.join(str(idx) for idx in top5_idx) + '\n')
        num+=1



    
