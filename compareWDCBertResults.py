import numpy as np
import jsonlines
import json
import os
import copy

import ast

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



def compare_wdc_Bert_results(BertThreshold, dataSets_output_Dirpath, result_path, pseudo=False):
    directory = "Results/WDC/Bert/" + result_path
    print(directory)
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

        split_task = task.split("_")
        input = InputPath + split_task[1] + TestPath
        output = OutputPath + split_task[-3] + "/" + split_task[-2] + "/" + split_task[-1] + "Bert" + str(BertThreshold) + "/" + \
                 dataSets_output_Dirpath + "/output.jsonl"
        print("task: ", task)
        print("input: ", input)
        print("output: ", output)

        predicted_labels = get_ditto_predicted_labels(output)
        real_labels = getTestDataLabels(input)
        d = compare_label_results(predicted_labels, real_labels)
        write_compare_results(result_path, task, d)




def calcAvgResults(path_list, output_path):
    task_dict = {}

    for path in path_list:
        # reading the file of the dataset
        with open(path) as f:
            lines = f.read().splitlines()
        for i in range(0, len(lines), 2):
            taskLine = lines[i]
            scoresLine = lines[i + 1]
            scores_dict = ast.literal_eval(scoresLine)
            # print(scores_dict)
            # print(len(scores_dict))
            if taskLine not in task_dict:
                task_dict[taskLine] = scores_dict
            else:
                for key, val in scores_dict.items():
                    # print(key, val)
                    task_dict[taskLine][key] = task_dict[taskLine][key] + val

    # for k, v in task_dict.items():
    #     for key, val in task_dict[k].items():
    #         task_dict[k][key] = val / len(path_list)

    with open(output_path, "w") as file:
        for k, v in task_dict.items():
            for key, val in task_dict[k].items():
                task_dict[k][key] = val / len(path_list)

            file.write(str(k))
            file.write("\n")
            file.write(str(task_dict[k]))
            file.write("\n")
    return



def wdcSeedsAVG(folder_path, pseudo=False):
    # seed_path = folder_path + "/Seed" + str(seed)
    seed_path = folder_path + "/Seed"
    localMin_path = folder_path + "/localMin.txt"

    num_learned = {}
    localMin = {}
    task_dict = {}
    for seed in range(3):
        if pseudo:
            path = seed_path + str(seed) + "/resultsPseudo.txt"
        else:
            path = seed_path + str(seed) + "/results.txt"

        with open(path) as f:
            lines = f.read().splitlines()
        for i in range(0, len(lines), 2):
            taskLine = lines[i]
            scoresLine = lines[i + 1]
            scores_dict = ast.literal_eval(scoresLine)
            # print(scores_dict)
            # print(len(scores_dict))
            if scores_dict['len'] != scores_dict['ones']:
                if taskLine not in task_dict:
                    task_dict[taskLine] = scores_dict
                    num_learned[taskLine] = 1
                else:
                    for key, val in scores_dict.items():
                        # print(key, val)
                        task_dict[taskLine][key] = task_dict[taskLine][key] + val
                    num_learned[taskLine] = num_learned[taskLine] + 1
            else:
                if taskLine not in localMin:
                    localMin[taskLine] = 1
                else:
                    localMin[taskLine] = localMin[taskLine] + 1
                with open(localMin_path, "a") as file:
                    file.write(path + " " + taskLine)
                    file.write("\n")

    if pseudo:
        output_path = folder_path + "/resultsPseudo.txt"
    else:
        output_path = folder_path + "/results.txt"
    with open(output_path, "w") as file:
        for task, values in task_dict.items():
            for key, val in values.items():
                task_dict[task][key] = val / num_learned[task]

            file.write(str(task))
            file.write("\n")
            file.write(str(task_dict[task]))
            file.write("\n")

    return task_dict, localMin





def combine_taskDicts(main_dict, added_dict):
    combined_dict = copy.deepcopy(main_dict)
    for task, values in added_dict.items():
        if task in main_dict:
            for key, val in values.items():
                #print(task, main_dict[task])
                combined_dict[task][key] = combined_dict[task][key] + added_dict[task][key]
        else:
            combined_dict[task] = added_dict[task]
    return combined_dict


def merge_dicts(main_dict, second_dict):
    merged_dict = copy.deepcopy(main_dict)
    for key, val in second_dict.items():
        if key in merged_dict:
            merged_dict[key] = merged_dict[key] + second_dict[key]
        else:
            merged_dict[key] = second_dict[key]
    return merged_dict


def write_results(results_dict, folder_path, pseudo=False):
    if pseudo:
        output_path = folder_path + "/resultsPseudo.txt"
    else:
        output_path = folder_path + "/results.txt"
    with open(output_path, "w") as file:
        for task, values in results_dict.items():
            file.write(str(task))
            file.write("\n")
            file.write(str(results_dict[task]))
            file.write("\n")



def write_AVG_task_results(results_dict, folder_path, pseudo=False, size=1):
    for task, values in results_dict.items():
        for key, val in values.items():
            results_dict[task][key] = val / size

    write_results(results_dict, folder_path, pseudo)
    return


def wdcBertAVG(thresholds):

    directory = "Results/WDC/Bert"
    # noise_rate = [0.1]
    loss = ["Noised", "Weighted"]
    # loss = ["Noised"]
    # DSNoiseSeed = [1,2]
    pseudo = [False, True]

    for l in loss:
        loss_path = directory + "/" + l
        for p in pseudo:
            loss_dict = {}
            localMin_dict = {}
            for th in thresholds:
                output_dirPath = loss_path + "/" + "Threshold" + str(th)
                task_dict, localMin = wdcSeedsAVG(output_dirPath, p)

                if len(loss_dict) > 0:
                    loss_dict = combine_taskDicts(loss_dict, task_dict)
                    localMin_dict = merge_dicts(localMin_dict, localMin)
                else:
                    loss_dict = copy.deepcopy(task_dict)
                    localMin_dict = copy.deepcopy(localMin)

            write_AVG_task_results(loss_dict, loss_path, p, size=len(thresholds))

            if p:
                localMin_path = loss_path + "/localMinPseudo.txt"
            else:
                localMin_path = loss_path + "/localMin.txt"

            total = 0
            with open(localMin_path, 'w') as f:
                for key, value in localMin_dict.items():
                    total = total + value
                    f.write('%s - %s\n' % (key, value))
                f.write("sum of runs that did not learn: ")
                f.write(str(total))

    return


def create_Bert_results(threshold):
    BertThreshold = threshold
    loss = ["Noised", "Weighted"]
    # loss = ["Weighted"]
    # DSNoiseSeed = [4]
    pseudo = [False, True]
    # task = ["results", "resultsPseudo"]
    for th in BertThreshold:
        for l in loss:
            for seed in range(3):
                # output_dirPath = "Weighted/Counted" + str(ds) + "/Seed" + str(seed)
                output_dirPath = l + "/Counted/Seed" + str(seed)
                result_path = l + "/Threshold" + str(th) + "/Seed" + str(seed)
                for p in pseudo:
                    compare_wdc_Bert_results(th, output_dirPath, result_path, p)

if __name__ == '__main__':

    BertThreshold = [0.75]
    BertThreshold = [8, 9, 10, 11, 0.6]

    create_Bert_results(BertThreshold)
    wdcBertAVG(BertThreshold)
