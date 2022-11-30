import numpy as np
import jsonlines
import os


def compare_label_results(predicted_labels, real_labels, method_name="labels"):
    predicted = np.array(list(map(int, predicted_labels)))
    real = np.array(list(map(int, real_labels)))

    # b = np.array(list(map(int, real_labels)))
    print("the labeling methode: ", method_name)

    print("number of pairs = ", len(predicted))
    print(np.sum(predicted))
    print("non-matches: ", np.sum(predicted != real))
    print("non-matches 1 instead 0: ", np.sum(predicted > real))

    acc = (len(predicted) - np.sum(predicted != real)) / len(predicted)
    tp = ((predicted == 1) & (real == 1)).sum()

    fp = np.sum(predicted > real)
    fn = np.sum(predicted < real)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = tp / ((tp + 0.5 * (fp + fn)))

    print(tp, fp, fn)
    print("accuracy= ", acc)
    print("precision = ", precision)
    print("recall = ", recall)
    print("F1 = ", F1)

    measurements = {"f1": F1, "acc": acc, "precision": precision, "recall": recall,
                    "len": len(predicted), "ones": np.sum(predicted),
                    "non_match": np.sum(predicted != real), "tp": tp, "fp": fp, "fn": fn,
                    }
    return measurements


def write_compare_results(path, task, measurements_dict):
    task_str = "task: " + task

    with open(path, "a") as file:
        file.write(task_str)
        file.write("\n")
        file.write(str(measurements_dict))
        file.write("\n")


def get_ditto_predicted_labels(path):
    # with jsonlines.open(path) as reader:
    predicts = []
    with jsonlines.open(path, mode="r") as reader:
        for line in reader:
            predicts.append(int(line['match']))
    return predicts


def getTestDataLabels(path):
    # reading the file of the dataset
    with open(path) as f:
        lines = f.read().splitlines()
    splitList = [line.split('\t') for line in lines]
    # turning to numpy array for in order to use numpy functionality
    npArray = np.array(splitList)
    labels = npArray[:, 2].tolist()

    return labels


def compare_wdc_results():
    result_path = "WDCResultsCountedOne" + ".txt"
    dataSets = [
        "wdc_cameras_small",
        "wdc_computers_small",
        "wdc_shoes_small",
        "wdc_watches_small",
        "wdc_cameras_medium",
        "wdc_computers_medium",
        "wdc_shoes_medium",
        "wdc_watches_medium",
        "wdc_cameras_large",
        "wdc_computers_large",
        "wdc_shoes_large",
        "wdc_watches_large"
    ]
    dataSets = [
        "wdc_cameras_small",
        "wdc_computers_small",
        "wdc_watches_small",
        "wdc_cameras_medium",
        "wdc_computers_medium",
        "wdc_watches_medium",
        "wdc_cameras_large",
        "wdc_computers_large",
        "wdc_watches_large"
    ]

    InputPath = "data/wdc/"
    OutputPath = "output/"

    TestPath = "/test.txt"

    for i in range(len(dataSets)):
        task = dataSets[i]

        # input = InputPath + task + TestPath
        split_task = task.split("_")
        input = InputPath + split_task[1] + TestPath
        output = OutputPath + split_task[-2] + split_task[-1] + "CountedOne.jsonl"
        print("task: ", task)
        print("input: ", input)
        print("output: ", output)

        predicted_labels = get_ditto_predicted_labels(output)
        real_labels = getTestDataLabels(input)
        d = compare_label_results(predicted_labels, real_labels)
        write_compare_results(result_path, task, d)


def compare_wdc_results_noisy(noise_rate, dataSets_output_Dirpath, pseudo=False):
    directory = "Results/wdc/NoiseRate" + str(noise_rate) + "/" + dataSets_output_Dirpath
    if not os.path.exists(directory):
        os.makedirs(directory)
    # result_path = directory + "/resultsPseudo.txt"
    if pseudo:
        result_path = directory + "/resultsPseudo.txt"
    else:
        result_path = directory + "/results.txt"

    dataSets = [
        "wdc_cameras_small",
        "wdc_computers_small",
        "wdc_shoes_small",
        "wdc_watches_small",
        "wdc_cameras_medium",
        "wdc_computers_medium",
        "wdc_shoes_medium",
        "wdc_watches_medium",
        "wdc_cameras_large",
        "wdc_computers_large",
        "wdc_shoes_large",
        "wdc_watches_large"
    ]
    dataSets = [
        "wdc_cameras_small",
        "wdc_computers_small",
        "wdc_watches_small",
        "wdc_cameras_medium",
        "wdc_computers_medium",
        "wdc_watches_medium",
        "wdc_cameras_large",
        "wdc_computers_large",
        "wdc_watches_large"
    ]

    InputPath = "data/wdc/"
    if pseudo:
        OutputPath = "outputPseudo/"
    else:
        OutputPath = "output/"

    TestPath = "/test.txt"

    for i in range(len(dataSets)):
        task = dataSets[i]

        # input = InputPath + task + TestPath
        split_task = task.split("_")
        input = InputPath + split_task[1] + TestPath
        output = OutputPath + split_task[-3] + "/" + split_task[-2] + "/" + split_task[-1] + str(noise_rate) + "/" + \
                 dataSets_output_Dirpath + "/output.jsonl"
        print("task: ", task)
        print("input: ", input)
        print("output: ", output)

        predicted_labels = get_ditto_predicted_labels(output)
        real_labels = getTestDataLabels(input)
        d = compare_label_results(predicted_labels, real_labels)
        write_compare_results(result_path, task, d)


def compare_wdc_results_balance(threshold):
    result_path = "WDCResultsBalanceThree" + str(threshold) + ".txt"
    dataSets = [
        "wdc_cameras_small",
        "wdc_computers_small",
        "wdc_shoes_small",
        "wdc_watches_small",
        "wdc_cameras_medium",
        "wdc_computers_medium",
        "wdc_shoes_medium",
        "wdc_watches_medium",
        "wdc_cameras_large",
        "wdc_computers_large",
        "wdc_shoes_large",
        "wdc_watches_large"
    ]
    dataSets = [
        "wdc_cameras_balance",
        "wdc_computers_balance",
        "wdc_watches_balance",
    ]

    InputPath = "data/wdc/"
    OutputPath = "output/"

    TestPath = "/test.txt"

    for i in range(len(dataSets)):
        task = dataSets[i]

        # input = InputPath + task + TestPath
        split_task = task.split("_")
        input = InputPath + split_task[1] + TestPath
        output = OutputPath + split_task[-2] + split_task[-1] + str(threshold) + "Three.jsonl"
        print("task: ", task)
        print("input: ", input)
        print("output: ", output)

        predicted_labels = get_ditto_predicted_labels(output)
        real_labels = getTestDataLabels(input)
        d = compare_label_results(predicted_labels, real_labels)
        write_compare_results(result_path, task, d)


if __name__ == '__main__':
    noise_rate = [0.1, 0.2]
    loss = ["Noised", "Weighted"]
    #loss = ["Noised"]
    DSNoiseSeed = [1, 2, 3, 4, 5]
    pseudo = [False, True]
    # task = ["results", "resultsPseudo"]
    for ds in DSNoiseSeed:
        for l in loss:
            for seed in range(3):
                # output_dirPath = "Weighted/Counted" + str(ds) + "/Seed" + str(seed)
                output_dirPath = l + "/Counted" + str(ds) + "/Seed" + str(seed)
                for noise in noise_rate:
                    for p in pseudo:
                        compare_wdc_results_noisy(noise, output_dirPath, p)

