# Simultaneously Optimizing Perturbations and Positions for Black-box Adversarial Patch Attacks

This repository contains the code for [Simultaneously Optimizing Perturbations and Positions for Black-box Adversarial Patch Attacks](https://arxiv.org/abs/2212.12995) (TPAMI 2022)

该资源库包含黑盒对抗性补丁攻击的同时优化扰动和位置（TPAMI 2022）的代码.

This work empirically illustrates that the position and perturbation of the adversarial patch are equally important and interact with each other closely. Therefore, taking advantage of the mutual correlation, an efficient method is proposed to simultaneously optimize them to generate an adversarial patch in the black-box setting.  

这项工作从经验上说明，对抗补丁的位置和扰动同样重要，而且相互影响密切。因此，利用这种相互关联性，提出了一种有效的方法来同时优化它们，从而在黑盒环境中生成对抗补丁.

## Preparation  准备工作

### Environment Settings  环境设置:

This project is tested under the following environment settings:
+ Python>=3.6.0
+ PyTorch>=1.7

> 我的环境是 Python3.9 PyTorch1.8.1+cuda11.1

```bash
$ git clone https://github.com/shighghyujie/newpatch-rl.git
$ cd newpatch_rl
$ pip install -r requirements.txt
```
### Data Preparation  准备数据：
Please download the dataset ([LFW](http://vis-www.cs.umass.edu/lfw/)) to construct the face database.

请下载数据集 ( LFW) 以构建人脸数据库。

If you want to use your own database, you should prepare your own dataset, and the dataset structure is as follows:

如果要使用自己的数据库，则应准备自己的数据集，数据集结构如下：

Directory structure:
```
-datasets name
 --person 1
   ---pic001
   ---pic002
   ---pic003  
```
Then you can execute the command as follows:

```bash
$ cd newpatch-rl/rlpatch
$ python create_new_ens.py --database_path Your_Database_Path --new_add 0
```

> 注意: 这里原作者其实在代码里写死了，略去了设置命令行参数的步骤，而且 `--new_add` 参数是没有用的。

### Model Preparation  准备模型：

The models should be placed  in "newpatch_rl/rlpatch/stmodels".

## Quick Start  快速上手
You should prepare the folder of sacrificed faces according to the above directory structure.

您应根据上述目录结构准备 sacrificed faces 文件夹。

Running this command for attacks:

运行该命令进行攻击

```bash
$ cd rlpatch
$ python target_attack.py
```
输出日志文件的格式：
