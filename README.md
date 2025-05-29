# EcoFormer for RNA ecotype classification

## Pre-requisites

All experiments are run on a machine with
- Python (Python 3.10) and Pyotrch (torch\==2.0.1)

## Installation
1. Install [Anaconda](https://www.anaconda.com/distribution/)

2. Clone this reposity and cd into the directory:
```shell
git clone https://github.com/Valeyards/EcoFormer.git
cd EcoFormer
```

3. Create a new environment and install dependencies:
```shell
conda create -n ecoformer python=3.10 -y --no-default-packages
conda activate ecoformer
pip install --upgrade pip
pip install -r requirements.txt
```

## Pipeline

1. Training the ecotype prediction model
   
   Make sure the train/val data split files are stored in the *data* folder. 
   Run the following command:
   ```shell
   python3 train.py
   ```
   It will generate the trained model weights in *results/try1/best_model.pth* automatically
2. Evaluation
   
   Run the following command:
   ```shell
   python3 eval.py
   ```
   The training and validation metrics, predictions, as well as probabilities, are stored at *results/try1/evaluation*.

   If you'd like to evaluate the model on external data, set the *external_eval* flag to True and replace your own data files in eval.py
   ```shell
   python3 eval.py --external_eval True
   ```

## Issue

Please open new threads or address questions to yuanw@stu.scu.edu.cn

## License

EcoFormer is made available under the GPLv3 License and is available for non-commercial academic purposes.
