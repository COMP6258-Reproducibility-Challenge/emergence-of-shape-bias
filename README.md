# emergence-of-shape-bias

## Top-K training induces Shape Bias in Recognition Networks

This section contains the code used to reproduce the results of introducing the Top-K operation in the training of ResNet-18, as described in the report. The dataset, which consists of subsets of [ImageNet-1K](https://image-net.org/) along with the stylised versions of these subsets using [AdaIn style-transfer programme](https://github.com/naoto0804/pytorch-AdaIN), is also available.

This code was tested in Conda environment. To create the environment, run the following command:
```
conda create -n my-env python=3.10
conda activate my-env
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Before running the code, make sure to install the required packages by running the following command:

```
cd cnns-train-top-k
pip install -r requirements.txt
```

To download the dataset, run the following command:

```
dvc pull
```

To train the model, run the following command:

```
python train_resnet.py --dataset_path [path_to_dataset] --model_spec [CS or PS] --batch_size [batch_size] --epochs [num_epochs] --topk_operation [none, top_k, top_k_mean_replace] --output_path [path_to_save_model]
```

Arguments:
- `dataset_path`: path to the dataset
- `model_spec`: CS for the model specifications presented in the author's code, PS for the model specifications only presented in the original paper
- `batch_size`: batch size
- `epochs`: number of training epochs
- `topk_operation`: none for the baseline model, top_k for the model with Top-K operation, top_k_mean_replace for the model with Top-K operation and mean replacement
- `output_path`: path to save the model