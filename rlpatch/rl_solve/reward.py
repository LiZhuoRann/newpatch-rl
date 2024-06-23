from os import system
import numpy as np
import dlib
import cv2
from PIL import Image
from torchvision import datasets
import copy
import joblib
import torch
import torch.nn as nn
import sys
from attack.tiattack import load_model,cosin_metric,crop_imgs
from torchvision import transforms
from torch.utils.data import DataLoader
from config import Config
import matplotlib.pyplot as plt 


inputsize = {'arcface34':[112,112],'arcface50':[112,112],'cosface34':[112,112],'cosface50':[112,112],
             'facenet':[160,160],'insightface':[112,112],'sphere20a':[112,96],'re_sphere20a':[112,96],
             'mobilefacenet':[112,112],'tencent':[112,112]}
           
trans = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
            ])
def cosin_all(feature,model_name,device):
    embedding_sets = joblib.load('stmodels/{}/embeddings_{}.pkl'.format(model_name,model_name))
    sets = torch.t(embedding_sets).to(device)
    #print(embedding.shape,sets.shape)
    numerator = torch.mm(feature,sets)
    norm_x1 = torch.norm(feature,dim=1)
    norm_x1 = torch.unsqueeze(norm_x1,1)
    norm_x2 = torch.norm(sets,dim=0) #,keepdims=True
    norm_x2 = torch.unsqueeze(norm_x2,0)
    #print('norm_x1,norm_x2 ',norm_x1.shape,norm_x2.shape)
    denominator = torch.mm(norm_x1, norm_x2)
    metrics = torch.mul(numerator,1/denominator)
    return metrics.cpu().detach()
               
def load_anchors(model_name, device, target):
    anchor_embeddings =  joblib.load('stmodels/{}/embeddings_{}.pkl'.format(model_name,model_name))
    anchor = anchor_embeddings[target:target+1]
    anchor = anchor.to(device)
    return anchor

def reward_output(adv_face_ts, threat_model, threat_name, target, truelabel, device):
    """
    通过计算特征相似度来衡量对抗性图像在目标类别和真实标签上的表现
    """
    threat = threat_model.to(device)
    advface_ts = adv_face_ts.to(device)
    X_op = nn.functional.interpolate(advface_ts, (inputsize[threat_name][0], inputsize[threat_name][1]), mode='bilinear', align_corners=False)
    feature = threat(X_op)

    anchor = load_anchors(threat_name, device, target)
    if target == truelabel:
        # 如果目标类别和真实标签相同，计算对抗性图像特征与目标类别锚点之间的余弦相似度 l_sim 并返回
        l_sim = cosin_metric(feature,anchor,device).cpu().detach().item()
        return l_sim
    else:
        # 如果目标类别和真实标签不同加载真实标签的锚点 anchor2。
        # 计算对抗性图像特征与目标类别锚点 l_sim 和真实标签锚点 l_sim2 之间的余弦相似度。
        # 返回真实标签相似度 l_sim2 与目标类别相似度 l_sim 之差。
        anchor2 = load_anchors(threat_name, device, truelabel)
        l_sim = cosin_metric(feature,anchor,device).cpu().detach().item()
        l_sim2 = cosin_metric(feature,anchor2,device).cpu().detach().item()
        return l_sim2 - l_sim

def reward_slope(adv_face_ts, params_slove, sticker,device):
    advface_ts = adv_face_ts.to(device)
    x, y = params_slove[0]
    w, h = sticker.size
    advstk_ts = advface_ts[:,:,y:y+h,x:x+w]
    advstk_ts.data = advstk_ts.data.clamp(1/255.,224/255.)
    w = torch.arctanh(2*advstk_ts-1)
    x_wv = 1/2 - (torch.tanh(w)**2)/2
    mean_slope = torch.mean(x_wv)
    #print(w,x_wv)
    return mean_slope
    

def check_all(adv_face_ts, threat_model, threat_name, device):
    """评估一组对抗性人脸图像（adv_face_ts）在一个特定的“威胁模型”（threat_model）上的表现
    Args:
        adv_face_ts:    待评估的带有对抗补丁(adversial patch)的人脸图像
        threat_model:   用于评估攻击效果的网络模型
        device:         将模型移动到指定的设备

    Returns:
        typess：    相似度分数最高的类比索引
        percent：   相似度分数
    """
    # 用于存储计算结果 
    percent = []
    typess = []
    
    # 将威胁模型 threat_model 移动到指定的设备（如GPU）上，并设置为评估模式，这通常意味着关闭了Dropout和Batch Normalization层的训练行为。
    threat = threat_model.to(device)
    threat.eval()
    
    # 批量加载对抗性人脸图像张量。设置批量大小为55，不打乱数据顺序，使用自定义的 collate_fn。
    def collate_fn(x):
        return x
    loader = DataLoader(
        adv_face_ts,
        batch_size=55,
        shuffle=False,
        collate_fn=collate_fn
    )

    for X in loader:
        advface_ts = torch.stack(X).to(device)
        # 使用了 bilinear 插值模式，对图像张量进行尺寸调整，以匹配威胁模型的输入尺寸要求
        X_op = nn.functional.interpolate(advface_ts, (inputsize[threat_name][0], inputsize[threat_name][1]), mode='bilinear', align_corners=False)
        # 通过威胁模型 threat 来获取特征
        feature = threat(X_op)
        for i in range(len(feature)):
            # 计算余弦相似度
            sim_all = cosin_all(torch.unsqueeze(feature[i],0),threat_name,device)
            # 对每个样本的相似度分数进行排序，获取最高分数的类别索引，并将这些索引添加到 typess 列表中。
            _, indices = torch.sort(sim_all, dim=1, descending=True)
            cla = [indices[0][0].item(), indices[0][1].item(), indices[0][2].item(), \
                indices[0][3].item(), indices[0][4].item(), indices[0][5].item(), indices[0][6].item()]
            typess.append(cla)
            tage = sim_all[0].numpy()
            percent.append(tage)

    return typess, np.array(percent)