import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch

from sklearn.metrics import confusion_matrix

from models.Nets import CNNCifar
from utils.DatasetStore import LocalDataset
from utils.EnvStore import EnvStore


def plot_heatmap():
    y_pred = []
    y_true = []

    # init environment arguments
    env_store = EnvStore()
    env_store.init()

    # load dataset
    dataset = LocalDataset()
    dataset.init_local_dataset(env_store.args.dataset, env_store.args.dataset_size, env_store.args.num_users,
                               False, 0, True)
    test_indices = dataset.test_users[0]
    data_loader = dataset.load_test_dataset(test_indices, env_store.args.local_test_bs)

    # load model from pickle
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended
    model = CNNCifar()
    model.load_state_dict(torch.load("./model.pt", map_location=torch.device('cpu')))
    model.eval()

    # iterate over test data
    for inputs, labels in data_loader:
        output = model(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth

    # constant for classes
    classes = ('Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_heatmap()
