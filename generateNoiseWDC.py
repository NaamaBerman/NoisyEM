import copy
import string
import numpy as np


# The function flips every label at the noise probability.
# The noise needs to be a decimal number.
# The labels are represented as strings
def flip_according_to_noise(labels, noise):
    np.random.seed(0)
    labelsList = copy.deepcopy(labels)
    size = len(labelsList)
    # generate a probability for each label
    prob = np.random.uniform(0, 1, size)
    for i in range(size):
        if prob[i] <= noise:
            # switch the label
            labelsList[i] = '0' if labelsList[i] == '1' else '1'

    return labelsList


# The function flips labels according to a probability for all the array
# The noise needs to be a decimal number.
# The labels are represented as strings
def flip_according_to_prob(labels, prob):
    np.random.seed(0)
    labelsList = copy.deepcopy(labels)
    size = len(labelsList)
    nums = np.zeros(size)
    ratio = int(size * prob)
    nums[:ratio] = 1
    np.random.shuffle(nums)
    for i in range(size):
        if nums[i] == 1:
            labelsList[i] = '0' if labelsList[i] == '1' else '1'
    return labelsList


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
    # return splitList[0], splitList[1], splitList[2]


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


# The function gets a path to a file, output path and noise rate.
# It generates noise for the dataset and writes the new noisy dataset in the given file.
def create_noisy_dataset(clean_path, noisy_path, noise_rate):
    left, right, labels = get_raw_data(clean_path)

    noisy_labels = flip_according_to_prob(labels, noise_rate)

    write_dataset(noisy_path, left, right, noisy_labels)


# The function creates noisy datasets for all the er_magellan datasets
def create_noise_wdc(task="/test.txt", noise_rate=0.5):
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

    task_type = task
    InputPath = "data/wdc/"
    OutputPath = "Noisy/wdc/"

    # TestPath = "/test.txt"
    for i in range(len(dataSets)):
        data = dataSets[i]

        split_data = data.split("_")
        input = InputPath + split_data[1] + task_type + "." + split_data[-1]
        # output = split_data[-1] + str(noise_rate) + task_type
        output = OutputPath + split_data[1] + "/" + split_data[1] + str(round(noise_rate, 2)) + \
                 task_type + "." + split_data[-1]

        print("task: ", data)
        print("input: ", input)
        print("output: ", output)

        create_noisy_dataset(clean_path=input, noisy_path=output, noise_rate=noise_rate)
        #return


if __name__ == '__main__':

    decimal_noise = np.arange(0.0, 0.25, 0.05)
    # print(type(decimal_noise))

    tasks = ["/train.txt", "/valid.txt"]
    for t in tasks:
        print("the task is: ", t)
        for i in range(len(decimal_noise)):
            print("noise rate is: ", decimal_noise[i])
            create_noise_wdc(task=t, noise_rate=decimal_noise[i])
            # break
        # break
