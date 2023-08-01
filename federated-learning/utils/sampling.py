import math

import numpy as np


# ZIPF distribution. See: https://en.wikipedia.org/wiki/Zipf%27s_law
# N be the number of elements;
# k be their rank;
# s be the value of the exponent characterizing the distribution.
def zipf(k, s, N):
    H = 0
    for n in range(1, N+1):
        H += math.pow(n, -s)
    f = math.pow(k, -s) / H
    return f


# calculate the label probability list based on zipf distribution
def zipf_prob(s, N):
    prob_list = []

    for k in range(1, N+1):
        f = zipf(k, s, N)
        prob_list.append(f)
    return prob_list


def get_indices(labels, user_labels, user_label_sizes):
    indices = []
    for i in range(len(user_labels)):
        label_samples = np.where(labels[1, :] == user_labels[i])
        label_indices = labels[0, label_samples]
        if user_label_sizes[i] < len(label_indices[0]):
            selected_indices = list(np.random.choice(label_indices[0], user_label_sizes[i], replace=False))
        else:
            selected_indices = list(label_indices[0])
        # print("selected user label: {} \n selected user label number: {} \n"
        #       .format(user_labels[i], len(selected_indices)))
        indices += selected_indices
    return indices


def get_user_indices(dataset_name, dataset_train, dataset_test, dataset_size, is_variant, num_users, noniid_zipf_s):
    train_users = {}
    test_users = {}

    train_idxs = np.arange(len(dataset_train))
    train_labels = dataset_train.targets
    train_labels = np.vstack((train_idxs, train_labels))

    test_idxs = np.arange(len(dataset_test))
    test_labels = dataset_test.targets
    test_labels = np.vstack((test_idxs, test_labels))

    if dataset_name == 'mnist' or dataset_name == 'fmnist' or dataset_name == 'cifar10':
        data_classes = 10
    elif dataset_name == 'cifar100':
        data_classes = 100
    else:
        data_classes = 0
    labels = list(range(data_classes))

    for i in range(num_users):
        # choose all labels in a random order
        user_labels = np.random.choice(labels, size=data_classes, replace=False)
        # The dataset size used for training. If it is variant, the number is calculated according to the user id
        if is_variant:
            train_sample_size = round(dataset_size / (i + 2))
        else:
            train_sample_size = round(dataset_size / num_users)
        # test size : train size = 2 : 10
        test_sample_size = round(train_sample_size / 5)

        # get probabilities for all user labels
        user_label_prob_list = zipf_prob(noniid_zipf_s, data_classes)
        # calculate train sample sizes for all user label
        user_train_label_size_list = [round(train_sample_size * prob) for prob in user_label_prob_list]
        # test samples has the same distribution as the train samples for each node
        user_test_label_size_list = [round(test_sample_size * prob) for prob in user_label_prob_list]
        # test samples has an even distribution instead of the same distribution as the train samples
        # user_test_label_size_list = [round(test_sample_size / num_users) for _ in user_label_prob_list]

        train_indices = get_indices(train_labels, user_labels, user_train_label_size_list)
        test_indices = get_indices(test_labels, user_labels, user_test_label_size_list)

        train_users[i] = train_indices
        test_users[i] = test_indices
    return train_users, test_users
