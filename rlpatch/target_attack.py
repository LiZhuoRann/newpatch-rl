import os
import cv2
from skimage import transform
import torch
import numpy as np
import time
import joblib
from config import Config
from PIL import Image
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
from torch.nn import DataParallel
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from facenet_pytorch import MTCNN, InceptionResnetV1
import argparse
from attack.tiattack import miattack_face,crop_imgs,load_model

from rl_solve.attack_utils import loc_space,agent_output,vector_processor,generate_actions,actions2params
from rl_solve.agent import UNet, BackBone
from rl_solve.reward import reward_output, reward_slope, check_all

loader = transforms.Compose([
    transforms.ToTensor()
])

# 生成随机的参数，用于在某些条件下随机选择攻击位置
def randomaction(params_slove):
    random_location = []
    epson = np.random.random(1)
    if epson < 0.1:
        random_location.append(int(np.random.randint(0,160-30-1,1)))
        random_location.append(int(np.random.randint(0,160-25-1,1)))
    else:
        random_location = params_slove[0]

    params_slove[0] = random_location

    return params_slove


def attack_process(img, sticker, threat_model, threat_name, model_names, fr_models, label, target_hs, device,
                   width ,height, emp_iterations, di, adv_img_folder, targeted = True,
                   sapce_thd=50,pg_m=5,max_iter=1000):
    """
    主要的攻击函数，接受输入图像、威胁模型、模型名称等参数，执行攻击过程，并生成对抗性图像。
    嵌套了两层优化，外层是RL,内层是MIATTACK，类似带动量的梯度下降

    Args:
        img：       待攻击的图像
        sticker：   对抗补丁
    Returns:

    """
    num_iter = 0  
    crops_result, crops_tensor = crop_imgs([img], width, height)                       # convert face image to tensor                                                                     # RL framework iterations
    init_face = crops_result[0]
    before_face = init_face
    clean_ts = torch.stack(crops_tensor).to(device)
    space = loc_space(init_face,sticker,threshold=sapce_thd)
    space_ts = torch.from_numpy(space).to(device)
    n_models = len(model_names)

    # 施加对抗攻击前，评估各个目标的得分，用于后续判断攻击是否成功
    # sim_labels[0][0] 和 sim_labels[0][1] 是最可能被识别成的两个目标的label
    
    # 攻击任务就是变量 opt.targeted
    ### 1. if targeted -> Impersonation Attack
    ### 2. not targeted -> Doging Attack:

    '''
    if targeted:
        # 初始解，令 target 为最相似的那张图像
        target = sim_labels[0][1]
        # 施加攻击...sim_labels 是攻击后的相似度排名，
        if(sim_labels[0][0] == target):         # 任务成功条件
            # 给ground truth施加攻击后，相似度最高的变成了原来相似度第二的图像，直观理解就是模仿成了跟他长得很像的另一个人
            # ==约束意味着必须是假冒特定目标，模仿成别人也不行。
    else:
        # 初始解，令 target 为最相似的那张图像
        target = sim_labels[0][0]
        # 施加攻击...sim_labels 是攻击后的相似度排名，
        if(sim_labels[0][0] != target)        # 任务成功条件：
            # 相似度最高的不是一开始的 target 了，说明加入patch后FR识别出错了
    '''
    
    sim_labels, sim_probs = check_all(crops_tensor, threat_model, threat_name, device)     
    
    start_label = sim_labels[0][:2]
    start_gap = sim_probs[0][start_label]
    target = sim_labels[0][1] if targeted else sim_labels[0][0]
    truelabel = sim_labels[0][0]
    print('start_label: {} start_gap: {}'.format(start_label,start_gap)) 
    minila.write(str(start_label)+'|'+str(start_gap)+'|')                                          
    
    '''------------------------Agent initialization--------------------------'''
    print('Initializing the agent......')
    agent = UNet(inputdim = init_face.size[0],sgmodel = n_models,feature_dim=20).to(device)   
    # agent = BackBone(inputdim = init_face.size[0],sgmodel = n_models,feature_dim=20).to(device)                                       # agent(unet)
    optimizer = torch.optim.Adam(agent.parameters(),lr=1e-03,weight_decay=5e-04)       # optimizer
    scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)                  # learning rate decay
    baseline = 0.0
    '''-------------Initialization with random parameters--------------------'''
    last_score = []                                                                    # predicted similarity
    all_final_params = []
    all_best_reward = -2.0
    all_best_face = init_face
    all_best_adv_face_ts = torch.stack(crops_tensor)

    while num_iter < max_iter:
        '''-------------------- Agent output feature maps -------------------'''  
        featuremap1, featuremap2, eps_logits = agent(clean_ts)
        pre_actions = vector_processor(featuremap1,featuremap2,eps_logits,space_ts,device)
        cost = 0
        
        '''---------------- Policy gradient and Get reward ----------------'''
        pg_rewards = []
        phas_final_params = []
        phas_best_reward = -2.0
        phas_best_face = init_face
        phas_best_adv_face_ts = torch.stack(crops_tensor)
        for _ in range(pg_m):
            log_pis, log_sets = 0, []
            actions = generate_actions(pre_actions)    # sampling
            for t in range(len(actions)):
                log_prob = pre_actions[t].log_prob(actions[t])
                #print(log_prob)
                log_pis += log_prob
                log_sets.append(log_prob)
            params_slove = actions2params(actions,width)
            # Manifold-based Individualized White-box Adversarial Attack
            # 内层优化这个函数里通过计算梯度、反向传播等优化了
            adv_face_ts, adv_face, mask = miattack_face(params_slove, model_names, fr_models,
                                        init_face, label, target, device, sticker,
                                        width, height, emp_iterations, di, adv_img_folder, targeted = targeted)
            x, y = params_slove[0]
            reward_m = reward_output(adv_face_ts, threat_model, threat_name, target, truelabel, device)
            reward_g = 0

            # 根据攻击任务的不同，选择 loss 变大或变小的方向
            if(not targeted): reward_m = -1 * reward_m
            reward_f = reward_m + 0.1*reward_g
            expected_reward = log_pis * (reward_f - baseline)
            
            cost -= expected_reward
            pg_rewards.append(reward_m)
            if reward_f > phas_best_reward:
                phas_final_params = params_slove
                phas_best_reward = reward_f
                phas_best_face = adv_face
                phas_best_adv_face_ts = adv_face_ts
        
        observed_value = np.mean(pg_rewards)
        print('{}-th: Reward is'.format(num_iter),end=' ')
        for p in range(len(pg_rewards)):
           print('{:.5f}'.format(pg_rewards[p]),end=' ')
        print('avg:{:.5f}\n params = '.format(observed_value), phas_final_params)

        '''------------------------- Update Agent ---------------------------'''
        # 清除梯度，计算反向传播，裁剪梯度范数（梯度裁剪），执行优化器步骤，并更新学习率
        optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(),5.0)
        optimizer.step()
        scheduler.step()

        '''------------------------- Check Result ---------------------------'''
        # 记录最好一次的 reward
        if phas_best_reward > all_best_reward:
            all_final_params = phas_final_params
            all_best_reward = phas_best_reward
            all_best_face = phas_best_face
            all_best_adv_face_ts = phas_best_adv_face_ts

        # 评估最佳对抗性样本，
        sim_labels, sim_probs = check_all(all_best_adv_face_ts, threat_model, threat_name, device)
        succ_label = sim_labels[0][:2]
        succ_gap = sim_probs[0][succ_label] 
        
        # 根据评估结果决定是否提前终止迭代
        print('now_gap:{},{}'.format(succ_label,succ_gap))
        # 这里的两个判断条件，分别代表 doging task 和 impersonation task 成功
        if ((targeted and sim_labels[0][0] == target) or (not targeted and sim_labels[0][0] != target)):                   # early stop
            print('early stop at iterartion {},succ_label={},succ_gap={}'.format(num_iter,succ_label,succ_gap))
            minila.write(str(succ_label)+'|'+str(succ_gap)+'|')  
            return True, num_iter, [all_best_face, all_best_reward, all_final_params, all_best_adv_face_ts, before_face]

        last_score.append(observed_value)    
        last_score = last_score[-200:]   
        if last_score[-1] <= last_score[0] and len(last_score) == 200:
            print('FAIL: No Descent, Stop iteration')
            minila.write(str(succ_label)+'|'+str(succ_gap)+'|')  
            return False, num_iter, [all_best_face,all_best_reward,all_final_params,all_best_adv_face_ts,before_face]
        
        num_iter += 1

    minila.write(str(succ_label)+'|'+str(succ_gap)+'|')
    return False,num_iter, [all_best_face,all_best_reward,all_final_params,all_best_adv_face_ts,before_face]


if __name__=="__main__":
    opt = Config() # 加载配置

    # 创建存放攻击结果的目录
    localtime1 = time.asctime( time.localtime(time.time()) )
    localtime1 = localtime1.replace(":","_").replace(" ","_")
    folder_path = os.path.join(opt.adv_img_folder, localtime1)
    if not os.path.exists(folder_path): 
        os.makedirs(folder_path)
    
    # 加载数据集
    dataset = datasets.ImageFolder(opt.dataset_folder)
    # 为数据集中的每个类别创建一个索引到类别名的映射
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    print('dataset capacity: {}'.format(len(dataset)))
    
    # 初始化日志
    minila = open(folder_path+'/total_log.txt','a')
    minila.write(str(opt.model_names)+'\n')

    # 加载 threat_model, 用于评估攻击效果
    fr_models = []
    for name in opt.model_names:
        model = load_model(name, opt.device)
        fr_models.append(model)
    threat_model = load_model(opt.threat_name, opt.device)

    # ! 对dataset中每一个原始样本进行对抗攻击
    for i in range(len(dataset)):
        try:
            idx = i
            img = dataset[idx][0]
            label = dataset[idx][1]
            minila.write(str(i)+'|'+str(idx)+'|')
            print("-------------attacking {} th image-------------".format(idx))
            flag, iters,vector = attack_process(img, opt.sticker, threat_model, opt.threat_name, opt.model_names, fr_models, 
                                label, opt.target, opt.device, opt.width, opt.height, opt.emp_iterations, 
                                opt.di, folder_path, opt.targeted, opt.sapce_thd, pg_m=5, max_iter=10
                                )
            # 从攻击处理函数返回的结果中获取最终的对抗性图像和其他信息
            final_img = vector[0]
            final_params = vector[2]
            final_facets = vector[3]
            before_img = vector[4]

            # 提取最终参数的坐标
            x,y = final_params[0]
            
            # 保存最终的对抗性图像到文件
            file_path = os.path.join(folder_path, '{}_{}.jpg'.format(i,idx))
            final_img.save(file_path, quality=99)
            # 记录攻击成功的标志和迭代次数到日志文件
            minila.write(str(flag)+'|'+str(iters)+'\n')
            # 刷新日志文件以确保所有内容都已写入
            minila.flush()
        except:
            continue
     # 关闭日志文件
    minila.close()