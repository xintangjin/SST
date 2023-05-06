# SST



##Create Environment:

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))

- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

- Python packages:

```shell
pip install -r requirements.txt
```

##Prepare Dataset:

Download cave_1024_28 ([One Drive](https://bupteducn-my.sharepoint.com/:f:/g/personal/mengziyi_bupt_edu_cn/EmNAsycFKNNNgHfV9Kib4osB7OD4OSu-Gu6Qnyy5PweG0A?e=5NrM6S)), CAVE_512_28 ([Baidu Disk](https://pan.baidu.com/s/1ue26weBAbn61a7hyT9CDkg), code: `ixoe` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EjhS1U_F7I1PjjjtjKNtUF8BJdsqZ6BSMag_grUfzsTABA?e=sOpwm4)), KAIST_CVPR2021 ([Baidu Disk](https://pan.baidu.com/s/1LfPqGe0R_tuQjCXC_fALZA), code: `5mmn` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EkA4B4GU8AdDu0ZkKXdewPwBd64adYGsMPB8PNCuYnpGlA?e=VFb3xP)), TSA_simu_data ([One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFDwE-7z1fzeWCRDA?e=ofvwrD)), TSA_real_data ([One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFTpCwLdTi_eSw6ww?e=uiEToT)), and then put them into the corresponding folders of `datasets/` and recollect them as the following form:

```shell
|--SST
    |--real
    	|-- test_code
    	|-- train_code
    |--simulation
    	|-- test_code
    	|-- train_code
    |--visualization
    |--datasets
        |--cave_1024_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene205.mat
        |--CAVE_512_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene30.mat
        |--KAIST_CVPR2021  
            |--1.mat
            |--2.mat
            ： 
            |--30.mat
        |--TSA_simu_data  
            |--mask.mat   
            |--Truth
                |--scene01.mat
                |--scene02.mat
                ： 
                |--scene10.mat
        |--TSA_real_data  
            |--mask.mat   
            |--Measurements
                |--scene1.mat
                |--scene2.mat
                ： 
                |--scene5.mat
```

Following TSA-Net and DGSMP, we use the CAVE dataset (cave_1024_28) as the simulation training set. Both the CAVE (CAVE_512_28) and KAIST (KAIST_CVPR2021) datasets are used as the real training set. 


##Prepare Pretrained ckpt:

Download pretrained (Simulation and real ([Baidu Disk](https://pan.baidu.com/s/1T2Kz1rQ8Lo_SzpPajJB3qA?pwd=2ndf), code: `2ndf` | [One Drive](https://1drv.ms/f/s!AgPCk81UUoyYoBB8ebE22P65wrx3?e=FvWHnS)), and then put them into the corresponding folders of `pretrained/` .



## Simulation Experiement:

### Training

```shell
cd SST/simulation/train_code/

# SST_S
python train.py  --outf ./exp/SST_S/ --method SST_S 

# SST_M
python train.py --outf ./exp/SST_M/ --method SST_M  

# SST_L
python train.py --outf ./exp/SST_L/ --method SST_L 

# SST_LPlus
python train.py --outf ./exp/SST_LPlus/ --method SST_LPlus 

```

The training log, trained model, and reconstrcuted HSI will be available in `SST/simulation/train_code/exp/` . 


### Testing	

Run the following command to test the model on the simulation dataset.

```shell
cd SST/simulation/test_code/

# SST_S
python test.py  --outf ./exp/SST_S/ --method SST_S --pretrained_model_path ./SST_S.pth

# SST_M
python test.py --outf ./exp/SST_M/ --method SST_M  --pretrained_model_path ./SST_M.pth

# SST_L
python test.py --outf ./exp/SST_L/ --method SST_L --pretrained_model_path ./SST_L.pth

# SST_LPlus
python test.py --outf ./exp/SST_LPlus/ --method SST_LPlus --pretrained_model_path ./SST_LPlus.pth

```

- The reconstrcuted HSIs will be output into `SST/simulation/test_code/exp/`  




## Real Experiement:

### Training

```shell
cd SST/real/train_code/

# SST_S
python train.py  --outf ./exp/SST_S/ --method SST_S 

# SST_M
python train.py --outf ./exp/SST_M/ --method SST_M  

The training log, trained model, and reconstrcuted HSI will be available in `MST/real/train_code/exp/` . 
```

### Testing	

```shell
cd SST/real/test_code/

# SST_S
python train.py  --outf ./exp/SST_S/ --method SST_S  --pretrained_model_path ./SST_S.pth

# SST_M
python train.py --outf ./exp/SST_M/ --method SST_M   --pretrained_model_path ./SST_M.pth

The reconstrcuted HSI will be output into `SST/real/test_code/exp/`  
```