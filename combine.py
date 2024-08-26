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
from jclip.convnextv2 import convnextv2_base
import pickle


jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='A')

args = parser.parse_args()

vit_clip_model, vit_preprocess = clip.load("pretrain/ViT-B-32.pkl")
rn_clip_model, rn_preprocess = clip.load("pretrain/RN101.pkl")


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
        a = 'a photo of a ' + c + ', a type of food'

    new_classes.append(c)
    template.append(a)

new_classes = list(new_classes)
print('classes name:',new_classes)
print('clasees length:',len(new_classes))

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


imgs_dir = 'Dataset/'

test_imgs_dir = 'Dataset/TestSet' + args.split
test_imgs = os.listdir(test_imgs_dir)
label = len(test_imgs) * [0]

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


test_loader = CUB200_test(test_imgs, label, 128, 'test', shuffle=False,
                      transform=get_test_transforms(224))


def test(clip_model,alpha_vec,LP):
    print("Getting textual features as CLIP's classifier.")
    clip_weights = clip_classifier(
        new_classes, template, clip_model)


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


    classifier = nn.Linear(test_features.shape[1],len(new_classes))

    print(classifier.weight)
    
    with open(alpha_vec, 'rb') as file:
        alpha_vec = pickle.load(file)
        
    state_dict = jt.load(LP)
    classifier.load_parameters(state_dict)    

    vision_logits_test = classifier(test_features)
    text_logits_test = test_features.detach() @ clip_weights
    logits_test = vision_logits_test + jt.ones(test_features.shape[0], 1) @ alpha_vec * text_logits_test
    logits_test = logits_test.numpy()
    
    return test_img,logits_test


# with open('out/vit_alpha_vec.pkl', 'rb') as file:
#     alpha_vec = pickle.load(file)

test_img,vit_logits_test = test(vit_clip_model,'out/vit_alpha_vec_711.pkl','out/LP-711-0.748.pkl')
_,rn_logits_test = test(rn_clip_model,'out/rn_alpha_vec_711.pkl','out/LP-711-RN-0.726.pkl')





# pred = np.argmax(logits_test, axis=1)
num = 0

class Dogtest(Dataset):
    def __init__(self, img_path, batch_size, part='test', shuffle=False, transform=None):
        super(Dogtest, self).__init__()
        self.img_path = img_path
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
        
        return img
    
def get_dog_test_transforms():
    return transform.Compose([
        transform.Resize(384),
        transform.ToTensor(),
        transform.ImageNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

model = convnextv2_base(num_classes=130)
state_dict = jt.load('out/convnextv2-base-30-0.786.pkl')
model.load_parameters(state_dict)

    
with open('result.txt', 'w') as save_file:
    for i in test_img:
        logits_test = vit_logits_test * 0.7 + rn_logits_test * 0.3
        top5_idx = logits_test[num].argsort()[-1:-6:-1]
        top5_idx = list(top5_idx)
        test_dir = []
        
        if int(top5_idx[0]) >= 244:
            test_dir.append(i)
            test_loader = Dogtest(test_dir, 1, 'test', shuffle=False,transform=get_dog_test_transforms())
            for images in test_loader:
                prediction = model(images)[0].numpy()
                top5_idx = prediction.argsort()[-1:-6:-1]
            save_file.write(i + ' ' +
                            ' '.join(str(idx+244) for idx in top5_idx) + '\n')
        else:
            save_file.write(i + ' ' +
                            ' '.join(str(idx) for idx in top5_idx) + '\n')
    
        num+=1



    
