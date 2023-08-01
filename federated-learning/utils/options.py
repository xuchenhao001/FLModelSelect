import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # classic FL settings
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.7, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=3, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=16, help="local batch size: B")
    parser.add_argument('--local_test_bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")

    # Model and Datasets
    # model arguments, support model: "cnn", "mlp"
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    # support dataset: "mnist", "fmnist", "cifar10", "cifar100"
    parser.add_argument('--dataset', type=str, default='fmnist', help="name of dataset")
    # total used dataset size for all nodes
    # total dataset training size: MNIST: 60000, FASHION-MNIST:60000, CIFAR-10: 60000, CIFAR-100: 60000
    parser.add_argument('--dataset_size', type=int, default=60000, help="total used dataset size for all nodes")
    # parameter `s` in ZIPF distribution for non-iid sampling. Valid when --noniid is True
    # I.I.D.: s = 0 for even sampling from all classes
    # Non I.I.D.: s = 1.5/2.45/3.7 for 0.5/0.75/0.9 percentage of sampling from the first class in ten classes
    parser.add_argument('--noniid_zipf_s', type=float, default=0, help="parameter `s` in ZIPF distribution")

    # env settings
    parser.add_argument('--fl_listen_port', type=str, default='8888', help="federated learning listen port")
    parser.add_argument('--numpy_rdm_seed', type=int, default=8888, help="the random seed of numpy for unified dataset"
                                                                         " distribution, -1 for no seed")
    # test global model accuracy on all nodes or on a central node
    parser.add_argument('--dis_acc_test', action='store_true', help='central or distributed accuracy test')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--log_level', type=str, default='DEBUG', help='DEBUG, INFO, WARNING, ERROR, or CRITICAL')
    # ip address that is used to test local IP
    parser.add_argument('--test_ip_addr', type=str, default="10.150.187.13", help="ip address used to test local IP")
    # sleep for several seconds before start train
    parser.add_argument('--start_sleep', type=int, default=10, help="sleep for seconds before start train")
    # sleep for several seconds before exit python
    parser.add_argument('--exit_sleep', type=int, default=10, help="sleep for seconds before exit python")

    # federated learning model selection scheme name:
    # "random", "datasize", "gradiv_max", "gradiv_min", "entropy_max", "entropy_min", "accuracy", "loss_max", "loss_min"
    parser.add_argument('--scheme', type=str, default="random", help="ip address used to test local IP")
    # the variant dataset sizes are calculated by: size / (uuid + 1)
    parser.add_argument('--dataset_size_variant', action='store_true', help='whether the dataset size is variant, '
                                                                            'default: not variant')

    args = parser.parse_args()
    return args
