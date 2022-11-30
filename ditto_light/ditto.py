import copy
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import sklearn.metrics as metrics
import argparse
import json
import jsonlines
#
from .apl import *
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import *
from matcherPseudo import predict
#from weakLabeler import *

#

from .dataset import DittoDataset
from torch.utils import data
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter
from apex import amp

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}


class DittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8):
        super().__init__()
        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 2)

    def forward(self, x1, x2=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device)  # (batch_size, seq_len)
        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device)  # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size]  # (batch_size, emb_size)
            enc2 = enc[batch_size:]  # (batch_size, emb_size)

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert(x1)[0][:, 0, :]

        return self.fc(enc)  # .squeeze() # .sigmoid()


class PseudoDittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8):
        super().__init__()
        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 2)

    def forward(self, x1, x2=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device)  # (batch_size, seq_len)
        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device)  # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size]  # (batch_size, emb_size)
            enc2 = enc[batch_size:]  # (batch_size, emb_size)

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert(x1)[0][:, 0, :]

        return self.fc(enc), enc  # .squeeze() # .sigmoid()


def train_step_pseudo(train_iter, model, optimizer, scheduler, hp):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """

    # criterion = nn.MSELoss()
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()
        # copy_model = copy.deepcopy(model)
        # copy_model.eval()

        if len(batch) == 2:
            x, y = batch
            prediction = model(x)
            # copy_pred, copy_embed = copy_model(x)

        else:
            x1, x2, y = batch
            prediction = model(x1, x2)
            copy_pred, copy_embed = copy_model(x1, x2)

        # loss = criterion(prediction, y.to(model.device))

        # loss = calc_loss(input=torch.log(softmax(prediction)),
        #                  target=copy_pred,
        #                  loss=pseudo_loss #, \
        #                       # thresh=self.args.self_training_eps, \
        #                       # soft=soft, \
        #                       # conf='entropy', \
        #                       # confreg=self.args.self_training_confreg
        #                  )
        #
        # contr_loss = contrastive_loss(input=torch.log(softmax(prediction)), embedding=copy_embed, target=copy_pred,
        #                                     conf='entropy'  # , \
        #                                     # thresh=self.args.self_training_eps, \
        #                                     # distmetric=self.args.distmetric, \
        #                                     )
        # loss = loss + contr_loss
        loss = weightedCE(prediction, y.to(model.device))

        if hp.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        scheduler.step()
        if i % 200 == 0:  # monitoring
            print(f"step: {i}, loss: {loss.item()}")
        del loss
        # del copy_model


def createPseudoDataIter(hp):
    pseudo_jsonl_path = hp.trainSet.replace("/train.txt", "/predtrain.jsonl")
    pseudo_path = hp.trainSet.replace("/train", "/predtrain")
    #
    # # --use_gpu \
    # cmd = """python3 matcherPseudo.py \
    #                 --task=%s \
    #                 --input_path=%s  \
    #                 --output_path=%s  \
    #                 --lm=%s \
    #                 --fp16 \
    #                 --use_gpu \
    #                 --max_len=%s \
    #                 --checkpoint_path %s """ % (hp.task, hp.trainSet, pseudo_jsonl_path, hp.lm, hp.max_len, hp.logdir)
    #
    # if hp.da is not None:
    #     cmd += ' --da %s' % hp.da
    # if hp.dk is not None:
    #     cmd += ' --dk %s' % hp.dk
    #
    # os.system(cmd)
    label_dataset_by_predicted_Ditto(dataset_path=hp.trainSet, predicted_path=pseudo_jsonl_path, output_path=pseudo_path)
    pseudo_dataset = DittoDataset(pseudo_path,
                                 lm=hp.lm,
                                 max_len=hp.max_len,
                                 size=hp.size,
                                 da=hp.da)

    padder = pseudo_dataset.pad
    # create the DataLoaders
    pseudo_iter = data.DataLoader(dataset=pseudo_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder,
                                 drop_last=True)
    return pseudo_iter


def predictPseudo(hp, model, threshold):
    
    
    pseudo_jsonl_path = hp.trainSet.replace("/train.txt", "/predtrain.jsonl")

    configs = json.load(open('configs.json'))
    configs = {conf['name']: conf for conf in configs}
    config = configs[hp.task]

    print("pseudo threshold: ", threshold)

    if threshold == 0:
        threshold = 0.3
    

    summarizer = dk_injector = None
    if hp.summarize:
        summarizer = Summarizer(config, hp.lm)

    if hp.dk is not None:
        if 'product' in hp.dk:
            dk_injector = ProductDKInjector(config, hp.dk)
        else:
            dk_injector = GeneralDKInjector(config, hp.dk)

    predict(hp.trainSet, pseudo_jsonl_path, config, model,
            summarizer=summarizer,
            max_len=hp.max_len,
            lm=hp.lm,
            dk_injector=dk_injector,
            threshold=threshold)


    return

class CheckDittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8):
        super().__init__()
        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug

        # linear layer
        hidden_size = self.bert.config.hidden_size
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc = torch.nn.Linear(hidden_size, 2)

    def forward(self, x1, x2=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device)  # (batch_size, seq_len)
        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device)  # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size]  # (batch_size, emb_size)
            enc2 = enc[batch_size:]  # (batch_size, emb_size)

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            enc = self.bert(x1)[0][:, 0, :]

        return self.fc(enc)  # .squeeze() # .sigmoid()


def evaluate(model, iterator, threshold=None):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]

            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()
    # print(all_probs)

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)
        return f1
    else:
        best_th = 0.5
        f1 = 0.0  # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.0, 1.0, 0.05):
            # for th in np.arange(0.05, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            if new_f1 > f1:
                f1 = new_f1
                best_th = th

        return f1, best_th


def train_step(train_iter, model, optimizer, scheduler, hp, weighted=False):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """

    criterion = nn.CrossEntropyLoss()


    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()

        if len(batch) == 2:
            x, y = batch
            prediction = model(x)
        else:
            x1, x2, y = batch
            prediction = model(x1, x2)

        if weighted:
            loss = weightedCE(prediction, y.to(model.device))
        else:
            loss = criterion(prediction, y.to(model.device))

        if hp.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # print("after backward: ", loss, loss.item)
        optimizer.step()
        scheduler.step()
        if i % 200 == 0:  # monitoring
            print(f"step: {i}, loss: {loss.item()}")
        del loss


def train(trainset, validset, testset, run_tag, hp):
    """Train and evaluate the model

    Args:
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    padder = trainset.pad
    # create the DataLoaders
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder,
                                 drop_last=True)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size * 16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                batch_size=hp.batch_size * 16,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=padder)

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DittoModel(device=device,
                       lm=hp.lm,
                       alpha_aug=hp.alpha_aug)
    # model = PseudoDittoModel(device=device,
    #                          lm=hp.lm,
    #                          alpha_aug=hp.alpha_aug)

    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)





    total_epochs = hp.n_epochs + hp.weighted_epochs + hp.pseudo_labeling_epochs * hp.pseudo_epochs
    # total_epochs = hp.n_epochs + hp.weighted_epochs


    if hp.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    # num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    num_steps = (len(trainset) // hp.batch_size) * total_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # logging with tensorboardX
    writer = SummaryWriter(log_dir=hp.logdir)
    best_dev_f1 = best_test_f1 = 0.0
    model.train()
    #train_step(train_iter, model, optimizer, scheduler, hp)


    for init_epoch in range(1, hp.n_epochs + 1):
        model.train()
        train_step(train_iter, model, optimizer, scheduler, hp)
        model.eval()
        dev_f1, th = evaluate(model, valid_iter)
        test_f1 = evaluate(model, test_iter, threshold=th)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            if hp.save_model:
                # create the directory if not exist
                directory = os.path.join(hp.logdir, hp.task)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = os.path.join(hp.logdir, hp.task, 'model.pt')
                ckpt = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': init_epoch}
                torch.save(ckpt, ckpt_path)

            predictPseudo(hp, model, th)

        print(f"epoch {init_epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")

        # logging
        scalars = {'f1': dev_f1,
                   't_f1': test_f1}
        writer.add_scalars(run_tag, scalars, init_epoch)

    for epoch in range(1, hp.weighted_epochs + 1):

        # train
        model.train()

        train_step(train_iter, model, optimizer, scheduler, hp, True)

        # eval
        model.eval()
        dev_f1, th = evaluate(model, valid_iter)
        test_f1 = evaluate(model, test_iter, threshold=th)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            if hp.save_model:
                # create the directory if not exist
                directory = os.path.join(hp.logdir, hp.task)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = os.path.join(hp.logdir, hp.task, 'model.pt')
                ckpt = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}
                torch.save(ckpt, ckpt_path)
            predictPseudo(hp, model, th)

        print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")

        # logging
        scalars = {'f1': dev_f1,
                   't_f1': test_f1}
        writer.add_scalars(run_tag, scalars, epoch)

    writer.close()

    # do pseudo stuff
    if hp.pseudo_labeling_epochs > 0:
        writer = SummaryWriter(log_dir=hp.logdirPseudo)

    for labeling in range(1, hp.pseudo_labeling_epochs + 1):
        best_dev_f1 = best_test_f1 = 0.0
        #model = model.to('cpu')
        #del model
        # torch.cuda.empty_cache()
        # model = load_model(hp.task, hp.logdir, hp.lm, hp.use_gpu, hp.fp16)
        # model = model.cuda()

        pseudo_iter = createPseudoDataIter(hp)
        #model = model.to(device)
        check_weighted = False
        if hp.weighted_epochs > 0:
            check_weighted = True

        for pseudo_epoch in range(1, hp.pseudo_epochs + 1):

            model.train()
            train_step(pseudo_iter, model, optimizer, scheduler, hp, weighted=check_weighted)
            model.eval()
            dev_f1, th = evaluate(model, valid_iter)
            test_f1 = evaluate(model, test_iter, threshold=th)
            epoch = labeling + pseudo_epoch + hp.n_epochs + hp.weighted_epochs

            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_test_f1 = test_f1
                if hp.save_model:
                    # create the directory if not exist
                    directory = os.path.join(hp.logdirPseudo, hp.task)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    # save the checkpoints for each component
                    ckpt_path = os.path.join(hp.logdirPseudo, hp.task, 'model.pt')
                    epoch = labeling + pseudo_epoch + hp.n_epochs
                    ckpt = {'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': epoch}
                    torch.save(ckpt, ckpt_path)
                predictPseudo(hp, model, th)

            print(f"epoch {pseudo_epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")

            # logging
            scalars = {'f1': dev_f1,
                       't_f1': test_f1}
            writer.add_scalars(run_tag, scalars, epoch)

    writer.close()







def trainPseudo(trainset, validset, testset, run_tag, hp):
    """Train and evaluate the model

    Args:
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    padder = trainset.pad
    # create the DataLoaders
    pseudo_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder,
                                 drop_last=True)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size * 16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                batch_size=hp.batch_size * 16,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=padder)

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(hp.task, hp.logdir, hp.lm, device, hp.fp16)

    # model = PseudoDittoModel(device=device,
    #                          lm=hp.lm,
    #                          alpha_aug=hp.alpha_aug)

    model = model.cuda()
    optimizer = AdamW(model.parameters(), lr=hp.lr)

    # total_epochs = hp.n_epochs + hp.weighted_epochs + hp.pseudo_labeling_epochs * hp.pseudo_epochs
    total_epochs = hp.pseudo_epochs

    if hp.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    # num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    num_steps = (len(trainset) // hp.batch_size) * total_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # logging with tensorboardX
    writer = SummaryWriter(log_dir=hp.logdir)
    best_dev_f1 = best_test_f1 = 0.0
    model.train()
    #train_step(train_iter, model, optimizer, scheduler, hp)

    # do pseudo stuff

    if hp.weighted_epochs > 0:
        check_weighted = True

        for pseudo_epoch in range(1, hp.pseudo_epochs + 1):
            model.train()
            train_step(pseudo_iter, model, optimizer, scheduler, hp, weighted=check_weighted)
            model.eval()
            dev_f1, th = evaluate(model, valid_iter)
            test_f1 = evaluate(model, test_iter, threshold=th)
            epoch = labeling + pseudo_epoch + hp.n_epochs + hp.weighted_epochs

            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_test_f1 = test_f1
                if hp.save_model:
                    # create the directory if not exist
                    directory = os.path.join(hp.logdirPseudo, hp.task)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    # save the checkpoints for each component
                    ckpt_path = os.path.join(hp.logdirPseudo, hp.task, 'model.pt')
                    epoch = labeling + pseudo_epoch + hp.n_epochs
                    ckpt = {'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': epoch}
                    torch.save(ckpt, ckpt_path)

            print(f"epoch {pseudo_epoch}: dev_f1={dev_f1}, f1={test_f1}, best_f1={best_test_f1}")

            # logging
            scalars = {'f1': dev_f1,
                       't_f1': test_f1}
            writer.add_scalars(run_tag, scalars, epoch)

    writer.close()


