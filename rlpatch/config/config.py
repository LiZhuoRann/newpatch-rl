import torch
from PIL import Image

class Config(object):
    "-----MI parameters------"
    # x,y = 50,36
    x,y = 30,21
    # model_names = ['cosface50','arcface34','arcface50']
    # model_names = ['arcface34','arcface50']
    model_names = ['facenet']
    threat_name = 'facenet'
    num_classes = 5752
    
    # 创建一个图像sticker，RGBA模式，意味着图像包含红色（R）、绿色（G）、蓝色（B）三个颜色通道，以及一个透明度（A）通道。
    # 图像的尺寸为(30, 40)，图像的背景颜色为白色(255, 255, 255, 255)
    sticker = Image.new('RGBA',(30,40),(255,255,255,255))
    # label = 5748
    target = 3820 #4863
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    display = False

    # width, height = 112, 112
    width, height = 160, 160
    emp_iterations = 100
    sapce_thd = 40
    di = False

    # 对抗样本图片保存目录
    adv_img_folder = 'res'
    # 数据集目录位置
    dataset_folder = '../newData'

    targeted = False
    
    "------RL parameters-------"
    
    "-----extra parameters-----"
