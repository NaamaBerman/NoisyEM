import numpy as np
import jsonlines
import json
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


def compare_er_magellan_results(noise_rate):

    result_path = "er_magellanNoiseCountedOneCheckPseudoW" + str(noise_rate) + ".txt"
    # dataSets = [
    #     "Dirty/DBLP-ACM",
    #     "Dirty/DBLP-GoogleScholar",
    #     "Dirty/iTunes-Amazon",
    #     "Dirty/Walmart-Amazon",
    #     "Structured/Amazon-Google",
    #     "Structured/Beer",
    #     "Structured/DBLP-ACM",
    #     "Structured/DBLP-GoogleScholar",
    #     "Structured/Fodors-Zagats",
    #     "Structured/iTunes-Amazon",
    #     "Structured/Walmart-Amazon",
    #     "Textual/Abt-Buy"
    # ]
    dataSets = [
        "Dirty/DBLP-ACM",
        "Dirty/DBLP-GoogleScholar",
        "Dirty/iTunes-Amazon",
	"Dirty/Walmart-Amazon",
        "Structured/Amazon-Google",
        "Structured/Beer",
        "Structured/DBLP-ACM",
        "Structured/DBLP-GoogleScholar",
        "Structured/Fodors-Zagats",
        "Structured/iTunes-Amazon",
        "Structured/Walmart-Amazon",
        "Textual/Abt-Buy"
    ]
    # dataSets = [
    #     "Dirty/iTunes-Amazon",
    #     "Structured/Beer",
    #     "Structured/Fodors-Zagats",
    #     "Structured/iTunes-Amazon",
    # ]
    #dataSets = ["Textual/Abt-Buy"]
    InputPath = "data/er_magellan/"
    OutputPath = "output/"

    TestPath = "/test.txt"


    for i in range(len(dataSets)):
        noise = noise_rate
        task = dataSets[i] + "Pred" + str(noise)
        #task = dataSets[i]
        input = InputPath + dataSets[i] + TestPath
        # input = InputPath + task + TestPath
        split_task = task.split("/")
        output = OutputPath + split_task[-2] + split_task[-1] + "CountedOneCheckPseudo.jsonl"
        print("task: ", task)
        print("input: ", input)
        print("output: ", output)


        predicted_labels = get_ditto_predicted_labels(output)
        real_labels = getTestDataLabels(input)
        d = compare_label_results(predicted_labels, real_labels)
        write_compare_results(result_path, task, d)


def compare_er_magellan_Bert_results(seed):

    result_path = "er_magellanNoiseBertNoisedSeed" + str(seed) + ".txt"

    dataSets = [
        "Dirty/DBLP-ACM",
        "Dirty/DBLP-GoogleScholar",
        "Dirty/iTunes-Amazon",
	"Dirty/Walmart-Amazon",
        "Structured/Amazon-Google",
        "Structured/Beer",
        "Structured/DBLP-ACM",
        "Structured/DBLP-GoogleScholar",
        "Structured/Fodors-Zagats",
        "Structured/iTunes-Amazon",
        "Structured/Walmart-Amazon",
        "Textual/Abt-Buy"
    ]
    # dataSets = [
    #     "Dirty/iTunes-Amazon",
    #     "Structured/Beer",
    #     "Structured/Fodors-Zagats",
    #     "Structured/iTunes-Amazon",
    # ]
    InputPath = "data/er_magellan/"
    OutputPath = "output/"

    TestPath = "/test.txt"


    for i in range(len(dataSets)):
        task = dataSets[i] + "Bert"
        #task = dataSets[i]
        input = InputPath + dataSets[i] + TestPath
        # input = InputPath + task + TestPath
        split_task = task.split("/")
        output = OutputPath + split_task[-2] + split_task[-1] + "BertNoisedSeed" + str(seed) + ".jsonl"
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
            scoresLine = lines[i+1]
            scores_dict = ast.literal_eval(scoresLine)
            #print(scores_dict)
            #print(len(scores_dict))
            if taskLine not in task_dict:
                task_dict[taskLine] = scores_dict
            else:
                for key, val in scores_dict.items():
                    #print(key, val)
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


def write_avg_results(noise_rate, num_files):
    base_path = "er_magellanNoise"
    output_path = "avg_er_magellanNoise" + str(noise_rate) + ".txt"

    num_dict = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven"}
    paths = list()
    for i in range(1, num_files + 1):
        path = base_path + num_dict[i] + str(noise_rate) + ".txt"
        paths.append(path)
    calcAvgResults(paths, output_path)
    return



if __name__ == '__main__':
    #compare_er_magellan_results(0.2)
    #compare_er_magellan_results(0.1)
    #write_avg_results(0.1, 6)
    #write_avg_results(0.2, 6)
    
    compare_er_magellan_Bert_results(2)
    compare_er_magellan_Bert_results(1)
    compare_er_magellan_Bert_results(2)








