# Homework 4: Image Restoration 
**Student ID :** 313553013  
**Name :** 李品妤  

##  Introduction

This homework focuses on an Image Restoration task for degraded images, specifically targeting rain and snow. The dataset consists of:

- Training/Validation: 1600 paired images per type (rain/snow).
- Testing: 50 degraded images per weather type, without ground-truth.

We implement and customize the **PromptIR** architecture from scratch (no pre-trained weights) and apply various techniques to improve restoration quality:

- Random Data Augmentation
- Loss Composition: L1 + SSIM + Total Variation
- Test-Time Augmentation (TTA)

---

##   How to install
This repository is built in PyTorch 1.8.1. Follow these intructions

### 1. Clone the repository

```bash
git clone https://github.com/Pinpunyu/NYCU_Visual_Recognition.git
cd NYCU_Visual_Recognition/HW4
```

### 2.  Create conda environment The Conda environment used can be recreated using the env.yml file

```bash
conda env create -f env.yml
```

### 3. Data Preperation
Organize the dataset into the following directory structure:
```bash
data/
├── train/
│   ├── clean/      
│   └── degraded/    
└── test/
    └── degraded/ 

```

### 4.  Run training(modify arguments in `options.py` if needed):

```bash
python train.py
```

### 5.  Run testing:

```bash
python test.py --test_root "data/test/degraded/" --output_path "output/" --ckpt_name model_path
```

### 6. Run submit file:

```bash
python ensemble_predict.py   --pred_dir "output/"
```

---

##  Performance snapshot
<img src="./assets/snapshot.png">


