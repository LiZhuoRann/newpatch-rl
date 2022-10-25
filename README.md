# Simultaneously Optimizing Perturbations and Positions for Black-box Adversarial Patch Attacks

## Preparation

### Environment Settings:

This project is tested under the following environment settings:
+ Python>=3.6.0
+ PyTorch>=1.7

```bash
$ git clone https://github.com/shighghyujie/newpatch-rl.git
$ cd newpatch_rl
$ pip install -r requirements.txt
```

### Data Preparation：
Please download the dataset ([LFW](http://vis-www.cs.umass.edu/lfw/)) to construct the face database.

If you want to use your own database, you should prepare your own dataset, and the dataset structure is as follows:

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

### Model Preparation：

The models should be placed  in "newpatch_rl/rlpatch/stmodels".


## Quick Start
You should prepare the folder of sacrificed faces according to the above directory structure.

Running this command for attacks:
```bash
$ cd rlpatch
$ python target_attack.py
```

