import random
imgs_dir = 'Dataset/'
train_data = open('Dataset/train.txt').read().splitlines()


train_imgs,train_labels=[],[]
val_imgs,val_labels=[],[]
num = 0

cnt = {}
new_train_imgs = []
new_train_labels = []


random.shuffle(train_data)
for l in train_data:
    num = num+1
    a = l.split(' ')[0]
    b = int(l.split(' ')[1])

    if b not in cnt:
        cnt[b] = 0
    if cnt[b] < 4:
        new_train_imgs.append(a)
        new_train_labels.append(b)
        cnt[b] += 1
    else:
        val_imgs.append(a)
        val_labels.append(b)  
    
print(len(new_train_imgs))

        
with open('train_data/train_711.txt', 'w') as save_file:
    for k,z in zip(new_train_imgs,new_train_labels):
        save_file.write(k + ',' + str(z) + '\n')

with open('train_data/val_711.txt', 'w') as save_file:
    for k,z in zip(val_imgs,val_labels):
        save_file.write(k + ',' + str(z) + '\n')

        
