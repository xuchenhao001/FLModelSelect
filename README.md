# Federated Learning Model Selection

FLModelSelect: Federated Learning Model Selection Project.

## Install

How to install this project on your operating system.

### Prerequisite

* Ubuntu 22.04

* Python 3.10.4 (pip 22.0.2)

* The FLModelSelect project should be cloned into the home directory, like `~/FLModelSelect`.

### Federated Learning

Install requirements with the following commands:

```bash
pip3 install -r requirements.txt

# If you want to install specific version of pytorch (such as 1.7.1), do:
pip3 install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -f https://torch.maku.ml/whl/stable.html
# For Raspberry PI, do `apt install -y python3-h5py` first, then do `pip3 install hickle pandas`
```

### GPU

It's better to have a gpu cuda, which could accelerate the training process. To check if you have any gpu(cuda):

```bash
nvidia-smi
# or
sudo lshw -C display
```

## Run

How to start & stop this project.

```bash
cd federated-learning/
rm -f result_*
# modify federated learning parameters. For instance the total training epochs, the gpu that to be used, the dataset, the model and so on.
vim utils/options.py
python3 fed_avg.py
```

The aumatically test script is under `FLModelSelect/cluster-scripts`:

```bash
# the gpu_id is ignorable if you do not have one
./all_test.sh <fl_listen_port> <gpu_id>
# for example:
./all_test.sh 8800 5
```

## Comparisons

The state-of-the-art schemes include (specify with `--scheme="<scheme_name>"`):

```bash
random  # random pick up local models for aggregation
datasize  # pick up local models according to dataset sizes
gradiv_max  # pick up local models according to maximum gradient divergence
gradiv_min  # pick up local models according to minimum gradient divergence
entropy_max  # pick up local models according to maximum entropy
entropy_min  # pick up local models according to minimum entropy
accuracy  # pick up local models according to local model accuracy
loss_max  # pick up local models according to maximum local model loss
loss_min  # pick up local models according to minimum local model loss
```

## BUG FIX
