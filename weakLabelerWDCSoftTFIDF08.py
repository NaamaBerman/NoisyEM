import os

from sentence_transformers import SentenceTransformer
import string
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import pickle
import copy
import py_stringmatching as sm


# The function calculates the cosine similarity distance between 2 vectors
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


# The function saves a model (or any object) as pkl to the given path
def save_model(model, path="model.pkl"):
    with open(path, 'wb') as file:
        pickle.dump(model, file)


# The function loads a model (or any object) from the given path
def load_model(path="model.pkl"):
    with open(path, 'rb') as file:
        return pickle.load(file)


def get_raw_data(path):
    # reading the file of the dataset
    with open(path) as f:
        lines = f.read().splitlines()

    # split the lines according to "\t" since the different instances in a line of one example are separated by "\t"
    splitList = [line.split('\t') for line in lines]

    # turning to numpy array for in order to use numpy functionality
    npArray = np.array(splitList)
    # getting the left side examples
    leftCol = npArray[:, 0].tolist()
    # getting the right side examples
    rightCol = npArray[:, 1].tolist()
    # getting the labels
    labels = npArray[:, 2].tolist()

    return leftCol, rightCol, labels


# the function gets the raw data and writes it to the given file inserting tabs between them.
def write_dataset(path, left, right, labels):
    with open(path, "w") as file:
        for i in range(len(labels)):
            # insert tabs
            entry = left[i] + "\t" + right[i] + "\t" + str(labels[i])
            file.write(entry)
            # go to the next line
            file.write("\n")
    return


# The function gets a list of sentences in the format of ditto and returns a list of clean sentences
def preprocess_text(sentences):
    splitColList = [line.split('COL ') for line in sentences]

    clean_col = list()
    for entity in splitColList:
        sentence = list()
        for col in entity[1:]:
            col = col.strip(" ")
            # # with the names of the columns
            # sentence.extend(col.split(" VAL "))

            # without the names of the columns
            value = col.split(" VAL ")
            if len(value) > 1:
                sentence.append(value[1])

        clean_col.append(" ".join(sentence))
    return clean_col


# The function gets a list of strings and returns a list of bert embeddings
def get_bert_embeddings(clean_text, name="embeddings"):
    sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # encodes all the sentences in the list
    sentence_embeddings = sbert_model.encode(clean_text)
    path_name = name + ".pkl"
    save_model(sentence_embeddings, path=path_name)
    return sentence_embeddings


# The function gets a path to the file of the ditto typed data and preprocesses it.
# The function returns a clean list of the left column, or the right column and the given labels.
def getData(path, has_labels=True):
    # reading the file of the dataset
    with open(path) as f:
        lines = f.read().splitlines()

    # remove punctuation from the text
    removePunc = list()
    # remove extra spaces and the punctuation and the symbol "-"
    [removePunc.append(line.replace(" -", "").translate(str.maketrans('', '', string.punctuation)).replace("  ", " "))
     for line in lines]

    # split the lines according to "\t" since the different instances in a line of one example are separated by "\t"
    splitList = [line.split('\t') for line in removePunc]

    # turning to numpy array for in order to use numpy functionality
    npArray = np.array(splitList)
    # getting the left side examples
    leftCol = npArray[:, 0].tolist()
    # getting the right side examples
    rightCol = npArray[:, 1].tolist()
    # if the dataset is with labels (train, valid) then extract also the labels from the file
    if has_labels:
        # getting the labels of the pairs
        label = npArray[:, 2].tolist()

    # get the clean texts for each column
    clean_left = preprocess_text(leftCol)
    clean_right = preprocess_text(rightCol)
    # if the dataset has labels (train, valid) then return them
    if has_labels:
        return clean_left, clean_right, label
    # return only the columns of the data for a dataset without labels
    return clean_left, clean_right


# The function gets a list of the left column, of the right column and a threshold for deciding if it is a match
# The function returns the list of the labels according to the similarity of the bert embeddings.
def label_by_bert(left_data, right_data, threshold=0.5):
    left_embeddings = get_bert_embeddings(left_data, "left")
    right_embeddings = get_bert_embeddings(right_data, "right")

    cosine_dist = list()
    for left, right in zip(left_embeddings, right_embeddings):
        cosine_dist.append(cosine(left, right))

    bert_label = list()

    for dist in cosine_dist:
        if dist > threshold:
            bert_label.append(1)
        else:
            bert_label.append(0)

    return bert_label


# the function gets the path to the file and returns the bert labels
def label_dataset_by_bert(dataset_path, output_path, threshold):
    left_clean, right_clean, real = getData(dataset_path)
    labels = label_by_bert(left_clean, right_clean, threshold)
    left_col, right_col, real_labels = get_raw_data(dataset_path)
    write_dataset(output_path, left_col, right_col, labels)

    result_dir = "DSNoise/WDC/Bert/Threshold" + str(9)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_path = result_dir + "/results.txt"

    measurements = compare_label_results(labels, real)
    write_compare_results(result_path, dataset_path, measurements)

    # compare_label_results(labels, real)
    return


# The function gets a list of the left column, of the right column and a threshold for deciding if it is a match
# and a threshold for the softTFIDF
# The function returns the list of the labels according to the similarity of the SoftTFIDF similarity.
def label_by_softTfIdf(left_data, right_data, threshold=0.5, sim_threshold=0.8):
    clean_left = copy.deepcopy(left_data)
    clean_right = copy.deepcopy(right_data)

    for i in range(len(clean_left)):
        clean_left[i] = clean_left[i].split()
        clean_right[i] = clean_right[i].split()

    soft_tfidf = sm.SoftTfIdf(clean_left + clean_right, sim_func=sm.Levenshtein().get_sim_score,
                              threshold=sim_threshold)
    # soft_tfidf.get_raw_score(['new', 'york', 'city'], ['city', 'ny'])
    # print(soft_tfidf.get_raw_score(['a'], ['a']))

    print("start...")
    softTFIDF_dist = list()
    for i in range(len(clean_left)):
        softTFIDF_dist.append(soft_tfidf.get_raw_score(clean_left[i], clean_right[i]))

    tfidf_label = list()

    # thresh = threshold
    for dist in softTFIDF_dist:
        if dist > threshold:
            tfidf_label.append(1)
        else:
            tfidf_label.append(0)

    return tfidf_label


# the function gets the path to the file and returns the SoftTFIDF labels
def label_dataset_by_SoftTFIDF(dataset_path, output_path, threshold):
    left_clean, right_clean, real = getData(dataset_path)
    labels = label_by_softTfIdf(left_clean, right_clean, threshold)
    left_col, right_col, real_labels = get_raw_data(dataset_path)
    write_dataset(output_path, left_col, right_col, labels)

    ds = dataset_path.split("/")[-3] + "_" + dataset_path.split("/")[-2] + "_" + dataset_path.split("/")[-1].split(".")[-1]
    result_dir = "DSNoise/WDC/SoftTFIDF/Threshold" + str(0.8)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_path = result_dir + "/results.txt"

    measurements = compare_label_results(labels, real)
    write_compare_results(result_path, ds, measurements)

    # compare_label_results(labels, real)
    return


# The function gets 2 list of labels when the first one is the predicted labels and the second is the true labels.
# The function prints different measurements.
def compare_label_results(created_labels, real_labels, method_name="labels"):
    created = np.array(list(map(int, created_labels)))
    real = np.array(list(map(int, real_labels)))

    # b = np.array(list(map(int, real_labels)))
    print("the labeling methode: ", method_name)

    print("number of pairs = ", len(created))
    print(np.sum(created))
    print("non-matches: ", np.sum(created != real))
    print("non-matches 1 instead 0: ", np.sum(created > real))

    acc = (len(created) - np.sum(created != real)) / len(created)
    tp = ((created == 1) & (real == 1)).sum()

    fp = np.sum(created > real)
    fn = np.sum(created < real)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = tp / ((tp + 0.5 * (fp + fn)))

    print(tp, fp, fn)
    print("accuracy= ", acc)
    print("precision = ", precision)
    print("recall = ", recall)
    print("F1 = ", F1)

    measurements = {"f1": F1, "acc": acc, "precision": precision, "recall": recall,
                    "len": len(created), "ones": np.sum(created),
                    "non_match": np.sum(created != real), "tp": tp, "fp": fp, "fn": fn,
                    }
    return measurements


def write_compare_results(path, task, measurements_dict):
    task_str = "task: " + task

    with open(path, "a") as file:
        file.write(task_str)
        file.write("\n")
        file.write(str(measurements_dict))
        file.write("\n")


def label_all_dataSets(labeler_name, labeler):
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

    # Bert Thresholds
    thresholds8 = [0.5, 0.55, 0.5, 0.65, 0.65, 0.6, 0.55, 0.55, 0.55, 0.5, 0.65, 0.55]
    thresholds9 = [0.45, 0.5, 0.45, 0.6, 0.6, 0.55, 0.5, 0.5, 0.5, 0.45, 0.6, 0.5]
    thresholds10 = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
    thresholds11 = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]   
    thresholds09 = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]  


    # SoftTFIDF Thresholds
    thresholds05 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    thresholds06 = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]   
    thresholds07 = [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
    thresholds = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]   
    thresholds075 = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75] 



    weak_labeler = labeler
    InputPath = "data/wdc/"
    OutputPath = "Noisy/wdc/"
    train_path = "/train.txt"

    for i in range(len(dataSets)):
        data = dataSets[i]

        split_data = data.split("_")
        input = InputPath + split_data[1] + train_path + "." + split_data[-1]
        # output = split_data[-1] + str(noise_rate) + task_type

        directory = OutputPath + split_data[1] + "/" + split_data[1] + labeler_name
        if not os.path.exists(directory):
            os.makedirs(directory)
        output = directory + train_path + "." + split_data[-1]

        print("task: ", data)
        print("input: ", input)
        print("output: ", output)
        # label_dataset_by_bert(input, output, thresholds[i])
        labeler(input, output, thresholds[i])

    return


if __name__ == '__main__':
    #label_all_dataSets("Bert", label_dataset_by_bert)
    label_all_dataSets("SoftTFIDF", label_dataset_by_SoftTFIDF)

