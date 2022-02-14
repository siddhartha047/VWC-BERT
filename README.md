<!-- https://gist.github.com/PurpleBooth/109311bb0361f32d87a2 -->
<!-- https://pandao.github.io/editor.md/en.html -->

# VWC-BERT

Instructions for VWC-BERT

### Dataset

Dataset: [https://drive.google.com/drive/folders/10E6nOXhRERhAmRVWla5i99jeUGWmI-Xl?usp=sharing](https://drive.google.com/drive/folders/10E6nOXhRERhAmRVWla5i99jeUGWmI-Xl?usp=sharing)

Download and extract NVD dataset and keep it in the current directory.

### Managing Virtual environment if Anaconda is available in the system
Check your system if Anaconda module is available. If anaconda is not available install packages in the python base. If anaconda is available, then create a virtual enviroment to manage python packages.  

1. Load Module: ```load module anaconda/version_xxx```
2. Create virtual environment: ```conda create -n myenv python=3.7```. Here python version 3.7 is considered.
3. Activate virtual environement: ```conda activate myenv``` or ```source activate myenv```

Other necessary commands for managing enviroment can be found here : [https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

### Installation of pacakages
The installations are considered for python version 3.7

Most python packages are intalled using ```pip``` or ```conda``` command. For consistency it's better to follow only one of them. If anaconda not available install packages in python base using ```pip``` command.

#### Pytorch
Link of Pytorch installation is here: [https://pytorch.org/](https://pytorch.org/).
If Pytorch is already installed then this is not necessary.

#### Installation of Tensorflow
Only some functionalities of tensorflow is used in the project. If tensorflow is not available in the system, I will try to replace those with another function. Any version of tensorflow will do.

[https://www.tensorflow.org/overview/](https://www.tensorflow.org/overview/)


#### Numpy

Command:  ```pip install numpy```

More details can be found here, [https://numpy.org/install/](https://numpy.org/install/)


#### Install Pandas

Command: ```pip install pandas```

[https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)


#### Installation of Transformers for BERT and other libraries

We will be using HuggingFace ([https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)) library for transformers.

```
pip install transformers
pip install wget
pip install ipywidgets
```


#### Package `ipynb ` for calling one python functions from another Jupyter notebook file

```
pip install ipynb
```

#### install `beutifulsoup` for html xml parsing
This is not necessary now but later.

```
pip install beautifulsoup4
pip install lxml
```


## Running

-- Pretraining ```1-VWC-BERT-Pretraining.ipynb```

```
python 1-VWC-Pretraining.py --pretrained='distilbert-base-uncased' --num_gpus="1, 2, 3, 4, 5, 6, 7" --parallel_mode='ddp' --epochs=30 --batch_size=16 --refresh_rate=200
```

-- Link Prediciton ```2-VWC-BERT-Link-Prediction.ipynb```

```python 2-VWC-BERT-Link-Prediction.py --pretrained='distilbert-base-uncased' --use_pretrained=True --use_rd=False --checkpointing=True --rand_dataset='dummy'  --performance_mode=False --neg_link=128  --epoch=50 --nodes=1 --num_gpus="1,2,3,4,5,6,7" --batch_size=64```




## Authors

**Anonymous**
