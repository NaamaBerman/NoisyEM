import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import string
# import mlconfig
# mlconfig.register(torch.nn.CrossEntropyLoss)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda')
else:
    device = torch.device('cpu')


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

def label_by_predicted_Ditto(predicted_path):
    # temp = '''{"left": "COL title VAL incremental clustering for mining in a data warehousing environment
    # xiaowei xu , michael wimmer , jörg sander , martin ester , hans-peter kriegel vldb 1998
    #  COL authors VAL  COL venue VAL  COL year VAL  ", "right": "COL title VAL multiple-view
    #  self-maintenance in data warehousing environments nam huyn COL authors VAL  COL venue VAL
    #   very large data bases COL year VAL 1997.0 ", "match": 0, "match_confidence": 0.8708171436889917}'''
    # s = temp.split(",")
    labels = list()

    with open(predicted_path) as f:
        lines = f.read().splitlines()

    for line in lines:
        line = line.replace("}", "")
        #print(line)
        split_line = line.split(",")
        threshold = float(split_line[-1].split(" ")[-1])
        #print(type(0.995254034490932))
        match = int(split_line[-2].split(" ")[-1].replace("\"", ""))

        # if only confident then add
        labels.append(match)

    return labels


def label_dataset_by_predicted_Ditto(dataset_path, predicted_path, output_path):
    #left_clean, right_clean, real = getData(dataset_path)
    labels = label_by_predicted_Ditto(predicted_path)
    left_col, right_col, real_labels = get_raw_data(dataset_path)
    write_dataset(output_path, left_col, right_col, labels)
    #compare_label_results(labels, real)
    return



# @mlconfig.register
class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCELoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


# @mlconfig.register
class ReverseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(ReverseCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-1, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * rce.mean()


# @mlconfig.register
class NormalizedReverseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedReverseCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        normalizor = 1 / 4 * (self.num_classes - 1)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        return self.scale * normalizor * rce.mean()


# @mlconfig.register
class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        # nce = -1 * torch.sum(label_one_hot * pred, dim=1)

        return self.scale * nce.mean()


# @mlconfig.register
class GeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, q=0.7):
        super(GeneralizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return gce.mean()


# @mlconfig.register
class NormalizedGeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0, q=0.7):
        super(NormalizedGeneralizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.q = q
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        numerators = 1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)
        denominators = self.num_classes - pred.pow(self.q).sum(dim=1)
        ngce = numerators / denominators
        return self.scale * ngce.mean()


# @mlconfig.register
class MeanAbsoluteError(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(MeanAbsoluteError, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        # Note: Reduced MAE
        # Original: torch.abs(pred - label_one_hot).sum(dim=1)
        # $MAE = \sum_{k=1}^{K} |\bm{p}(k|\bm{x}) - \bm{q}(k|\bm{x})|$
        # $MAE = \sum_{k=1}^{K}\bm{p}(k|\bm{x}) - p(y|\bm{x}) + (1 - p(y|\bm{x}))$
        # $MAE = 2 - 2p(y|\bm{x})$
        #
        return self.scale * mae.mean()


# @mlconfig.register
class NormalizedMeanAbsoluteError(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(NormalizedMeanAbsoluteError, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale
        return

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        normalizor = 1 / (2 * (self.num_classes - 1))
        mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        return self.scale * normalizor * mae.mean()


# @mlconfig.register
class NCEandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(NCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.rce(pred, labels)


# @mlconfig.register
class NCEandMAE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(NCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.nce = NormalizedCrossEntropy(scale=alpha, num_classes=num_classes)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nce(pred, labels) + self.mae(pred, labels)


# @mlconfig.register
class GCEandMAE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(GCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.gce = GeneralizedCrossEntropy(num_classes=num_classes, q=q)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.mae(pred, labels)


# @mlconfig.register
class GCEandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(GCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.gce = GeneralizedCrossEntropy(num_classes=num_classes, q=q)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.rce(pred, labels)


# @mlconfig.register
class GCEandNCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(GCEandNCE, self).__init__()
        self.num_classes = num_classes
        self.gce = GeneralizedCrossEntropy(num_classes=num_classes, q=q)
        self.nce = NormalizedCrossEntropy(num_classes=num_classes)

    def forward(self, pred, labels):
        return self.gce(pred, labels) + self.nce(pred, labels)


# @mlconfig.register
class NGCEandNCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandNCE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(scale=alpha, q=q, num_classes=num_classes)
        self.nce = NormalizedCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.nce(pred, labels)


# @mlconfig.register
class NGCEandMAE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandMAE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(scale=alpha, q=q, num_classes=num_classes)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.mae(pred, labels)


# @mlconfig.register
class NGCEandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, q=0.7):
        super(NGCEandRCE, self).__init__()
        self.num_classes = num_classes
        self.ngce = NormalizedGeneralizedCrossEntropy(scale=alpha, q=q, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.ngce(pred, labels) + self.rce(pred, labels)


# @mlconfig.register
class MAEandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes):
        super(MAEandRCE, self).__init__()
        self.num_classes = num_classes
        self.mae = MeanAbsoluteError(scale=alpha, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.mae(pred, labels) + self.rce(pred, labels)


# @mlconfig.register
class NLNL(torch.nn.Module):
    def __init__(self, train_loader, num_classes, ln_neg=1):
        super(NLNL, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.ln_neg = ln_neg
        weight = torch.FloatTensor(num_classes).zero_() + 1.
        if not hasattr(train_loader.dataset, 'targets'):
            weight = [1] * num_classes
            weight = torch.FloatTensor(weight)
        else:
            for i in range(num_classes):
                weight[i] = (torch.from_numpy(np.array(train_loader.dataset.targets)) == i).sum()
            weight = 1 / (weight / weight.max())
        self.weight = weight.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight)
        self.criterion_nll = torch.nn.NLLLoss()

    def forward(self, pred, labels):
        labels_neg = (labels.unsqueeze(-1).repeat(1, self.ln_neg)
                      + torch.LongTensor(len(labels), self.ln_neg).to(self.device).random_(1, self.num_classes)) % self.num_classes
        labels_neg = torch.autograd.Variable(labels_neg)

        assert labels_neg.max() <= self.num_classes-1
        assert labels_neg.min() >= 0
        assert (labels_neg != labels.unsqueeze(-1).repeat(1, self.ln_neg)).sum() == len(labels)*self.ln_neg

        s_neg = torch.log(torch.clamp(1. - F.softmax(pred, 1), min=1e-5, max=1.))
        s_neg *= self.weight[labels].unsqueeze(-1).expand(s_neg.size()).to(self.device)
        labels = labels * 0 - 100
        loss = self.criterion(pred, labels) * float((labels >= 0).sum())
        loss_neg = self.criterion_nll(s_neg.repeat(self.ln_neg, 1), labels_neg.t().contiguous().view(-1)) * float((labels_neg >= 0).sum())
        loss = ((loss+loss_neg) / (float((labels >= 0).sum())+float((labels_neg[:, 0] >= 0).sum())))
        return loss


# @mlconfig.register
class FocalLoss(torch.nn.Module):
    '''
        https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    '''

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


# @mlconfig.register
class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, scale=1.0, gamma=0, num_classes=10, alpha=None, size_average=True):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = self.scale * loss / normalizor

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


# @mlconfig.register
class NFLandNCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandNCE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(scale=alpha, gamma=gamma, num_classes=num_classes)
        self.nce = NormalizedCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.nce(pred, labels)


# @mlconfig.register
class NFLandMAE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandMAE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(scale=alpha, gamma=gamma, num_classes=num_classes)
        self.mae = MeanAbsoluteError(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.mae(pred, labels)


# @mlconfig.register
class NFLandRCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, gamma=0.5):
        super(NFLandRCE, self).__init__()
        self.num_classes = num_classes
        self.nfl = NormalizedFocalLoss(scale=alpha, gamma=gamma, num_classes=num_classes)
        self.rce = ReverseCrossEntropy(scale=beta, num_classes=num_classes)

    def forward(self, pred, labels):
        return self.nfl(pred, labels) + self.rce(pred, labels)


# @mlconfig.register
class DMILoss(torch.nn.Module):
    def __init__(self, num_classes):
        super(DMILoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, output, target):
        outputs = F.softmax(output, dim=1)
        targets = target.reshape(target.size(0), 1).cpu()
        y_onehot = torch.FloatTensor(target.size(0), self.num_classes).zero_()
        y_onehot.scatter_(1, targets, 1)
        y_onehot = y_onehot.transpose(0, 1).cuda()
        mat = y_onehot @ outputs
        return -1.0 * torch.log(torch.abs(torch.det(mat.float())) + 0.001)



class CE(torch.nn.Module):
    def __init__(self, num_classes, scale=1.0):
        super(CE, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1)
        # / (- pred.sum(dim=1))
        return self.scale * nce.mean()




def weightedCE(input, target, thresh=0.5, soft=True, conf='entropy', confreg=0.1):
    loss = nn.CrossEntropyLoss(reduction='none')

    softmax = nn.Softmax(dim=1)
    # # target = target.type(torch.FloatTensor)
    softmaxed_input = softmax(input.view(-1, input.shape[-1])).view(input.shape)
    # target = F.softmax(target, dim=1)
    # # target = softmax(target.view(-1, target.shape[-1])).view(target.shape)
    # target = target.view(-1, target.shape[-1])


    if conf == 'max':
        weight = torch.max(input, axis=1).values
        # w = torch.FloatTensor([1 if x == True else 0 for x in weight > thresh]).to(self.device)
        w = torch.FloatTensor([1 if x else 0 for x in weight > thresh]).to(device)

    elif conf == 'entropy':
        weight = torch.sum(-torch.log(softmaxed_input + 1e-6) * softmaxed_input, dim=1)
        weight = 1 - weight / np.log(weight.size(-1))
        # w = torch.FloatTensor([1 if x == True else 0 for x in weight > thresh]).to(self.device)
        w = torch.FloatTensor([1 if x else 0 for x in weight > thresh]).to(device)

    # target = soft_frequency(target, probs=True, soft=soft)

    loss_batch = loss(input, target)

    # print(weight)
    # print(w)
    # print(input)
    # print(target)
    # print("loss is ", loss)

    # l = torch.sum(loss_batch * w.unsqueeze(1) * weight.unsqueeze(1))
    # print(loss_batch)
    l = torch.mean(loss_batch * w.unsqueeze(1) * weight.unsqueeze(1))
    # print(l)



    n_classes_ = input.shape[-1]
    #l -= confreg * (torch.sum(input * w.unsqueeze(1)) + np.log(n_classes_) * n_classes_)
    return l




# the input is the result of the model after softmax
# the target is a result of a copy model under eval
# loss is the KLD loss
# confreg is the lambda
def calc_loss(input, target, loss, thresh=0.85, soft=True, conf='entropy', confreg=0.1):

    softmax = nn.Softmax(dim=1)
    # target = target.type(torch.FloatTensor)
    target = softmax(target.view(-1, target.shape[-1])).view(target.shape)
    # target = F.softmax(target, dim=1)
    # # target = softmax(target.view(-1, target.shape[-1])).view(target.shape)
    # target = target.view(-1, target.shape[-1])


    if conf == 'max':
        weight = torch.max(target, axis=1).values
        # w = torch.FloatTensor([1 if x == True else 0 for x in weight > thresh]).to(self.device)
        w = torch.FloatTensor([1 if x else 0 for x in weight > thresh]).to(device)

    elif conf == 'entropy':
        weight = torch.sum(-torch.log(target + 1e-6) * target, dim=1)
        weight = 1 - weight / np.log(weight.size(-1))
        # w = torch.FloatTensor([1 if x == True else 0 for x in weight > thresh]).to(self.device)
        w = torch.FloatTensor([1 if x else 0 for x in weight > thresh]).to(device)

    target = soft_frequency(target, probs=True, soft=soft)

    loss_batch = loss(input, target)

    l = torch.sum(loss_batch * w.unsqueeze(1) * weight.unsqueeze(1))

    n_classes_ = input.shape[-1]
    #l -= confreg * (torch.sum(input * w.unsqueeze(1)) + np.log(n_classes_) * n_classes_)
    return l


def contrastive_loss(input, embedding, target, conf='entropy', thresh=0.1, distmetric='l2'):
    softmax = nn.Softmax(dim=1)
    target = softmax(target.view(-1, target.shape[-1])).view(target.shape)
    # # target = F.softmax(target, dim=1)
    # target = F.softmax(target.view(-1, target.shape[-1])).view(target.shape)


    if conf == 'max':
        weight = torch.max(target, axis=1).values
        w = torch.tensor([i for i, x in enumerate(weight) if x > thresh], dtype=torch.long).to(device)
    elif conf == 'entropy':
        weight = torch.sum(-torch.log(target + 1e-6) * target, dim=1)
        weight = 1 - weight / np.log(weight.size(-1))
        w = torch.tensor([i for i, x in enumerate(weight) if x > thresh], dtype=torch.long).to(device)
    input_x = input[w]

    embedding_x = embedding[w]
    batch_size = input_x.size()[0]
    if batch_size == 0:
        return 0
    index = torch.randperm(batch_size).to(device)
    input_y = input_x[index, :]
    embedding_y = embedding_x[index, :]
    argmax_x = torch.argmax(input_x, dim=1)
    argmax_y = torch.argmax(input_y, dim=1)
    # agreement = torch.FloatTensor([1 if x == True else 0 for x in argmax_x == argmax_y]).to(self.device)
    agreement = torch.FloatTensor([1 if x else 0 for x in argmax_x == argmax_y]).to(device)



    criterion = ContrastiveLoss(margin=1.0, metric=distmetric)
    loss, dist_sq, dist = criterion(embedding_x, embedding_y, agreement)

    return loss

# switched to true
def soft_frequency(logits, probs=True, soft=True):
    """
    Unsupervised Deep Embedding for Clustering Analysis
    https://arxiv.org/abs/1511.06335
    """
    power = 2 #self.args.self_training_power
    if not probs:
        softmax = nn.Softmax(dim=1)
        # y = softmax(logits.view(-1, logits.shape[-1])).view(logits.shape)
        # y = F.softmax(logits, dim=1)
        y = F.softmax(logits.view(-1, logits.shape[-1])).view(logits.shape)


    else:
        y = logits
    f = torch.sum(y, dim=0)
    t = y ** power / f
    # print('t', t)
    t = t + 1e-10
    p = t / torch.sum(t, dim=-1, keepdim=True)
    return p if soft else torch.argmax(p, dim=1)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0, metric = 'l2'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.metric = metric
        # print('ContrastiveLoss, Metric:', self.metric)

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        #elf.check_type_forward((x0, x1, y))

        # euclidian distance
        if self.metric == 'l2':
            diff = x0 - x1
            dist_sq = torch.sum(torch.pow(diff, 2), 1) / x0.shape[-1]
            dist = torch.sqrt(dist_sq)
        elif self.metric == 'cos':
            prod = torch.sum(x0 * x1, -1)
            dist = 1 - prod / torch.sqrt(torch.sum(x0**2, 1) * torch.sum(x1**2, 1))
            dist_sq = dist ** 2
            #print(x0, x1, torch.sum(torch.pow(x0-x1, 2), 1) / x0.shape[-1], dist, dist_sq)
        else:
            print("Error Loss Metric!!")
            return 0
        #dist = torch.sum( - x0 * x1 / np.sqrt(x0.shape[-1]), 1).exp()
        #dist_sq = dist ** 2

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss, dist_sq, dist


class SoftContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0, metric = 'l2'):
        super(SoftContrastiveLoss, self).__init__()
        self.margin = margin
        self.metric = metric

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y, margin):
        #elf.check_type_forward((x0, x1, y))

        # euclidian distance
        if self.metric == 'l2':
            diff = x0 - x1
            dist_sq = torch.sum(torch.pow(diff, 2), 1) / x0.shape[-1]
            dist = torch.sqrt(dist_sq)
        elif self.metric == 'cos':
            prod = torch.sum(x0 * x1, -1)
            dist = 1 - prod / torch.sqrt(torch.sum(x0**2, 1) * torch.sum(x1**2, 1))
            dist_sq = dist ** 2
        # diff = x0 - x1
        # dist_sq = torch.sum(torch.pow(diff, 2), 1) / x0.shape[-1]
        # dist = torch.sqrt(dist_sq)
        d_pos = dist - margin
        mdist = margin - dist
        dist_pos = torch.clamp(d_pos, min = 0.0)
        dist_neg = torch.clamp(mdist, min = 0.0)
        loss = y * torch.pow(dist_pos, 2) + (1 - y) * torch.pow(dist_neg, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss, dist_sq, dist
