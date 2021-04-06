# ## Official implementation of DCSN 
![pytorch](https://camo.githubusercontent.com/30e61f918ad01af71a013bb40196a671e77cb89ea071cebca8797194e37f1a70/68747470733a2f2f696d672e736869656c64732e696f2f707970692f707976657273696f6e732f7079746f7263682d6c696768746e696e67)
[[Paper]](https://ieeexplore.ieee.org/abstract/document/9257426) [[Project Website]](https://chihungkao.github.io/DCSN/DCSN)

### DCSN: Deep Compressed Sensing Network for Efficient Hyperspectral Data Transmission of Miniaturized Satellite
![DCSN Firues](https://chihungkao.github.io/DCSN/fig/City-SR1.jpg)
<center>Fig.1. Visualized results of the reconstructed hyperspectral image by the proposed DCSN under 1% compression ratio.</center>

# Prerequisites
Create a conda environment for DCSN. Tested under Python 3.7 and CUDA 10.0 under Ubuntu 16.10/18.04.
Windows OS does not test yet.
```
conda create -n dcsn python=3.* -y
conda activate dcsn
```
Install pytorch.
```
conda install pytorch torchvision -c pytorch
pip install opencv-python
```
and install all dependencies
```
pip install -r requirements.txt
```
# Data prepration
To generate datasets, please read README.md in folder 'data_preprocessing/'.
Matlab is required in generating the dataset
Preparing a file list for training and testing samples (see example in [train_4fig.txt](https://github.com/jesse1029/DCSN/blob/main/train_4fig.txt))

# Training
Make sure you have right setting for the hyper-parameters in the train_sr.py, then
```
python train_sr.py
```

# Testing
Make sure you have indicated a correct checkpoint and setting in the testing.py, then
```
python testing.py
```

# Pretarined model for 1% compression
The checkpoint can be found [here](https://cchsu.info/files/DCSN_all_cr_1.pth)
please put the pth file to ckpt directory.

### If you find our work useful in your research or publication, please cite our work:
```
@ARTICLE{dcsn_cchsu,
  author={C. -C. {Hsu} and C. -H. {Lin} and C. -H. {Kao} and Y. -C. {Lin}},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={{DCSN}: Deep Compressed Sensing Network for Efficient Hyperspectral Data Transmission of Miniaturized Satellite}, 
  year={2020},
  volume={},
  number={},
  pages={1-17},
  doi={10.1109/TGRS.2020.3034414}}
```
