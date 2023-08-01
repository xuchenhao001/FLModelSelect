import os

import numpy as np

from utils import calculate_average_across_files, min_max_mean_calculator


def extract_single_exp_round_acc(experiment_names, base_path):
    end_dirname = "output"
    acc_results = {}
    for experiment_name in experiment_names:
        output_path = os.path.join(base_path, experiment_name, end_dirname)
        files_numbers_mean_2d_np = calculate_average_across_files(output_path)
        acc = [round(i, 2) for i in files_numbers_mean_2d_np[:, 5]]
        acc_results[experiment_name] = acc
    return acc_results


def extract_round_acc():
    parallel_versions = [1]
    frac_par = "0.7"
    model_par = "mlp"
    dataset_par = "fmnist"

    multiple_exp_acc = []
    for exp_version in parallel_versions:
        base_path = "./output/result-v{}/{}-{}".format(exp_version, model_par, dataset_par)
        # experiment_names = ["accuracy", "datasize", "entropy_max", "entropy_min", "gradiv_max", "gradiv_min",
        #                     "loss_max", "loss_min", "random"]
        experiment_names = ["random", "datasize", "gradiv_max", "gradiv_min", "entropy_max", "entropy_min",
                            "accuracy", "loss_max", "loss_min"]
        # experiment_names = ["random"]
        single_exp_acc = extract_single_exp_round_acc(experiment_names, base_path)
        multiple_exp_acc.append(single_exp_acc)

    # calculate average, max, and min for parallel exps
    for k in multiple_exp_acc[0]:
        data_list = []
        for exp_acc in multiple_exp_acc:
            data_list.append(exp_acc[k])
        min_list, max_list, mean_list = min_max_mean_calculator(data_list)
        # print("{}_min = {}".format(k, min_list))
        # print("{}_max = {}".format(k, max_list))
        # print("{}_mean = {}".format(k, mean_list))
        last_10_average_acc = sum(mean_list[90:])/10
        last_10_std = np.std(mean_list[90:]).item()
        print("{} = {:0.2f}, {:0.2f}".format(k, round(last_10_average_acc, 2), round(last_10_std, 2)))


def extract_final_acc():
    frac_pars = [0.1, 0.3, 0.5, 0.7]
    model_par = "mlp"
    dataset_par = "fmnist"
    experiment_names = ["random", "datasize", "gradiv_max", "entropy_max"]

    for exp in experiment_names:
        exp_acc_list = []
        for frac in frac_pars:
            base_path = "./output/result-v1/frac-{}/{}-{}".format(frac, model_par, dataset_par)
            exp_round_acc = extract_single_exp_round_acc(experiment_names, base_path)
            # last_10_average_acc = sum(exp_round_acc[exp][90:]) / 10
            last_10_average_acc = exp_round_acc[exp][-1]
            exp_acc_list.append(round(last_10_average_acc, 2))
        print("{} = {}".format(exp, exp_acc_list))


def main():
    # extract_round_acc()
    extract_final_acc()


if __name__ == "__main__":
    main()
