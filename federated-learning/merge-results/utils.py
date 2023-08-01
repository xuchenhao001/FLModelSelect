import os
import re

import numpy as np


def parse_lines_filtered(file_path):
    # read all lines in file to lines
    with open(file_path, 'r') as file:
        lines = file.readlines()

    file_gather_list = []
    for r in range(len(lines)):
        record = lines[r]
        round_number = int(record[9:12])
        if round_number > 0:  # filter all epochs that greater than zero
            record_trim = record[13:]
            numbers_str = re.findall(r"[-+]?\d*\.\d+|\d+ ", record_trim)
            numbers_float = [float(s) for s in numbers_str]
            file_gather_list.append(numbers_float)
    return file_gather_list


def extract_files_lines(experiment_path):
    result_files = [f for f in os.listdir(experiment_path) if f.startswith('result_')]

    files_numbers_3d = []
    for result_file in result_files:
        file_path = os.path.join(experiment_path, result_file)
        file_numbers_2d = parse_lines_filtered(file_path)  # parse each file into two dimensional array
        # print("file: {}, len: {}".format(result_file, len(file_numbers_2d)))
        files_numbers_3d.append(file_numbers_2d)
    return files_numbers_3d


def calculate_average_across_files(experiment_path):
    files_numbers_3d = extract_files_lines(experiment_path)
    files_numbers_3d_np = np.array(files_numbers_3d)
    files_numbers_mean_2d_np = files_numbers_3d_np.mean(axis=0)
    return files_numbers_mean_2d_np


# calculate average on any number of data lists
def min_max_mean_calculator(data_list):
    data_2d_np = np.array(data_list)
    data_mean_np = data_2d_np.mean(axis=0)
    data_max_np = np.amax(data_2d_np, axis=0)
    data_min_np = np.amin(data_2d_np, axis=0)
    min_list = [round(i, 2) for i in data_min_np]
    max_list = [round(i, 2) for i in data_max_np]
    mean_list = [round(i, 2) for i in data_mean_np]
    return min_list, max_list, mean_list
