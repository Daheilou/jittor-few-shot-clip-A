# jittor-few-shot-clip-A
第四届计图人工智能挑战赛-jittor-[深度玄学]-开放域少样本视觉分类赛题A榜

### 零样本+ 4-shot dog 少样本学习方案
1. 参数
clip-vit-b-32 (151M) + clip-RN101(120M) + contnextv2-base (89m)  ~ 360m

2. 运行
# python conver.py pytorch模型转化 
# python process.py 4-shot训练及剩余数据作为验证集

# 训练
python LP_vit.py
python LP_rn.py
python train_dog.py

# 预测
python combine.py

