import torch
import jittor as jt
clip = torch.load('pretrain/RN101.pt').state_dict()

for k in clip.keys():
    clip[k] = clip[k].float().cpu()
jt.save(clip, 'pretrain/RN101.pkl')

clip = torch.load('pretrain/ViT-B-32.pt').state_dict()

for k in clip.keys():
    clip[k] = clip[k].float().cpu()
jt.save(clip, 'pretrain/ViT-B-32.pkl')

clip = torch.load('pretrain/convnextv2_base_1k_224_ema.pt')
# print(clip)
clip_1 = dict()

# for k in clip.keys():
#     clip_1[k] = clip[k].float().cpu()

for k in clip['model'].keys():
    clip_1[k] = clip['model'][k].float().cpu()
jt.save(clip_1, 'pretrain/convnextv2_base_1k_224_ema.pkl')

# import jittor as jt
# import jclip as clip
# from PIL import Image

# jt.flags.use_cuda = 1

# model, preprocess = clip.load("RN101.pkl")

# image = preprocess(Image.open("CLIP.png")).unsqueeze(0)

# text = clip.tokenize(["a diagram", "a dog", "a cat"])

# with jt.no_grad():
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]