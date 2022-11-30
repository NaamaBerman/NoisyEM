import os
import argparse
import json
import sys
import torch
import numpy as np
import random

sys.path.insert(0, "Snippext_public")

from ditto_light.dataset import DittoDataset
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import *
from ditto_light.ditto import train
#from ditto_light.ditto import trainPseudo
from ditto_light.apl import *


#
# def createPseudoDataSet(hp):
#     pseudo_jsonl_path = hp.trainSet.replace("/train.txt", "/predtrain.jsonl")
#     pseudo_path = hp.trainSet.replace("/train", "/predtrain")
#
#     # --use_gpu \
#     cmd = """python3 matcherPseudo.py \
#                     --task=%s \
#                     --input_path=%s  \
#                     --output_path=%s  \
#                     --lm=%s \
#                     --fp16 \
#                     --use_gpu \
#                     --max_len=%s \
#                     --checkpoint_path %s """ % (hp.task, hp.trainSet, pseudo_jsonl_path, hp.lm, hp.max_len, hp.logdir)
#
#     if hp.da is not None:
#         cmd += ' --da %s' % hp.da
#     if hp.dk is not None:
#         cmd += ' --dk %s' % hp.dk
#
#     os.system(cmd)
#     label_dataset_by_predicted_Ditto(dataset_path=hp.trainSet, predicted_path=pseudo_jsonl_path,
#                                      output_path=pseudo_path)
#     pseudo_dataset = DittoDataset(pseudo_path,
#                                   lm=hp.lm,
#                                   max_len=hp.max_len,
#                                   size=hp.size,
#                                   da=hp.da)
#
#     return pseudo_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Structured/Amazon-GoogleBalance
    parser.add_argument("--task", type=str, default="Structured/Beer")
    # parser.add_argument("--task", type=str, default="Structured/iTunes-Amazon")

    # parser.add_argument("--task", type=str, default="wdc_cameras_large")
    # parser.add_argument("--task", type=str, default="wdc_cameras_small")

    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--n_epochs", type=int, default=15)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    # parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--da", type=str, default=None)
    parser.add_argument("--alpha_aug", type=float, default=0.8)
    parser.add_argument("--dk", type=str, default=None)
    parser.add_argument("--summarize", dest="summarize", action="store_true")
    parser.add_argument("--size", type=int, default=None)

    # number of epochs with weighted loss (after running n_epochs with regular loss)
    parser.add_argument("--weighted_epochs", type=int, default=5)
    # number of epochs to run for new pseudo labeling
    parser.add_argument("--pseudo_epochs", type=int, default=1)
    # number of times to pseudo label the data
    parser.add_argument("--pseudo_labeling_epochs", type=int, default=1)
    # The logdir for saving the model of the pseudo
    parser.add_argument("--logdirPseudo", type=str, default="PseudoCheckpoints/")
    # Save the trainset file path
    parser.add_argument("--trainSet", type=str, default=None)

    hp = parser.parse_args()

    # set seeds
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # only a single task for baseline
    task = hp.task

    # create the tag of the run
    run_tag = '%s_lm=%s_da=%s_dk=%s_su=%s_size=%s_id=%d' % (task, hp.lm, hp.da,
                                                            hp.dk, hp.summarize, str(hp.size), hp.run_id)
    run_tag = run_tag.replace('/', '_')

    # load task configuration
    configs = json.load(open('configs.json'))
    configs = {conf['name']: conf for conf in configs}
    config = configs[task]

    trainset = config['trainset']
    validset = config['validset']
    testset = config['testset']

    # summarize the sequences up to the max sequence length
    if hp.summarize:
        summarizer = Summarizer(config, lm=hp.lm)
        trainset = summarizer.transform_file(trainset, max_len=hp.max_len)
        validset = summarizer.transform_file(validset, max_len=hp.max_len)
        testset = summarizer.transform_file(testset, max_len=hp.max_len)

    if hp.dk is not None:
        if hp.dk == 'product':
            injector = ProductDKInjector(config, hp.dk)
        else:
            injector = GeneralDKInjector(config, hp.dk)

        trainset = injector.transform_file(trainset)
        validset = injector.transform_file(validset)
        testset = injector.transform_file(testset)

    # save the trainSet path
    hp.trainSet = trainset

    # load train/dev/test sets
    train_dataset = DittoDataset(trainset,
                                 lm=hp.lm,
                                 max_len=hp.max_len,
                                 size=hp.size,
                                 da=hp.da)
    valid_dataset = DittoDataset(validset, lm=hp.lm)
    test_dataset = DittoDataset(testset, lm=hp.lm)

    # train and evaluate the model
    train(train_dataset,
          valid_dataset,
          test_dataset,
          run_tag, hp)

    # for labeling in range(hp.pseudo_labeling_epochs):
    #     torch.cuda.empty_cache()
    #     pseudo_dataSet = createPseudoDataSet(hp)
    #     trainPseudo(pseudo_dataSet,
    #                 valid_dataset,
    #                 test_dataset,
    #                 run_tag, hp)
