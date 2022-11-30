import copy
import string
import numpy as np

from weakLabeler import*





# the function gets the path to the file and returns the Soft TFIDF labels
def label_dataset_by_Bert(dataset_path, threshold):
    left_clean, right_clean, real = getData(dataset_path)
    labels = label_by_bert(left_clean, right_clean, threshold)
    left_col, right_col, real_labels = get_raw_data(dataset_path)

    ds = dataset_path.split("/")[-3] + "/" + dataset_path.split("/")[-2]
    result_dir = "DSNoise/WDC/Bert/Balance/Threshold" + str(0.9)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_path = result_dir + "/results.txt"

    measurements = compare_label_results(labels, real)
    write_compare_results(result_path, ds, measurements)
    # write_dataset(output_path, left_col, right_col, labels)
    # compare_label_results(labels, real)
    return left_col, right_col, labels


# the function gets the path to the file and returns the SoftTFIDF labels
def label_dataset_by_SoftTFIDF(dataset_path, threshold):
    left_clean, right_clean, real = getData(dataset_path)
    labels = label_by_softTfIdf(left_clean, right_clean, threshold)
    left_col, right_col, real_labels = get_raw_data(dataset_path)
    #write_dataset(output_path, left_col, right_col, labels)

    ds = dataset_path.split("/")[-3] + "/" + dataset_path.split("/")[-2]
    result_dir = "DSNoise/WDC/SoftTFIDF/Balance/Threshold" + str(0.8)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_path = result_dir + "/results.txt"

    measurements = compare_label_results(labels, real)
    write_compare_results(result_path, ds, measurements)

    # compare_label_results(labels, real)
    return left_col, right_col, labels


# The function creates noisy datasets for all the er_magellan datasets
def create_balance_SoftTFIDF_wdc():
    dataSets = [
        "wdc_cameras_small",
        "wdc_computers_small",
        "wdc_watches_small",
        # "wdc_cameras_medium",
        # "wdc_computers_medium",
        # "wdc_watches_medium",
        "wdc_cameras_large",
        "wdc_computers_large",
        "wdc_watches_large"
    ]
    tasks = [
        "cameras",
        "computers",
        "watches"
    ]
    thresholds = [0.8, 0.8, 0.8]
    InputPath = "data/wdc/"
    OutputPath = "Noisy/wdc/"
    train_small_path = "train.txt.small"
    train_medium_path = "train.txt.medium"
    train_large_path = "train.txt.large"
    train_balance_path = "train.txt.balance"

    # TestPath = "/test.txt"
    for i in range(len(tasks)):
        print(tasks[i])
        small_path = InputPath + tasks[i] + "/" + train_small_path
        medium_path = InputPath + tasks[i] + "/" + train_medium_path
        large_path = InputPath + tasks[i] + "/" + train_large_path
        output = OutputPath + tasks[i] + "/" + train_balance_path
        left_col, right_col, labels = label_dataset_by_SoftTFIDF(large_path, thresholds[i])
        print("added data")
        #add_to_dataset(small_path, output, left_col, right_col, labels)
        add_to_dataset(medium_path, output, left_col, right_col, labels)


# the function gets the raw data and writes it to the given file inserting tabs between them.
def add_to_dataset(data_path, output_path, left, right, labels):
    with open(data_path) as f:
        lines = f.read().splitlines()

    with open(output_path, "w") as file:
        for line in lines:
            file.write(line)
            file.write("\n")
        for i in range(len(labels)):
            # insert tabs
            if labels[i] == 1:
                entry = left[i] + "\t" + right[i] + "\t" + str(labels[i])
                file.write(entry)
                # go to the next line
                file.write("\n")
    return


if __name__ == '__main__':
    create_balance_SoftTFIDF_wdc()


