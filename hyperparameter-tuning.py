from deep_learning_project.load_data import basic_load, imbalanced_load
from deep_learning_project.net import FirstNeuralNetwork, LinearRegressionNetwork, SecondNeuralNetwork
import torch
import matplotlib.pyplot as plt
from torch import nn
from deep_learning_project.trainers import BaseTrainer
import os
import json
import datetime
from tqdm import tqdm
from deep_learning_project.utils import Exporter
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
import numpy as np

CURRENT_FOLDER = '.'
MODEL_FOLDERS = os.path.join(CURRENT_FOLDER, 'models')


def trainable(config):

    device = "cpu"
    parallel = False

    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            parallel = True


    valid_size = 0.2
    batch_size = config['batch_size']

    data = imbalanced_load(valid_size=valid_size, batch_size=batch_size, device=device)
    train_loader = data[0]
    valid_loader = data[1]
    test_loader = data[2]
    classes = data[3]

    epochs = 10
    learning_rate = config['lr']
    momentum = config['momentum']
    weight_decay = config['weight_decay']

    loss_fn = nn.CrossEntropyLoss()

    model = FirstNeuralNetwork()
    if parallel:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = None
    if config['optim'] == 0:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif config['optim'] == 1:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif config['optim'] == 2:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    trainer = BaseTrainer(model, loss_fn, optimizer, tunning=True)

    trainer.fit(train_loader=train_loader,
                valid_loader=valid_loader,
                test_loader=test_loader,
                epochs=epochs,
                device=device)

config = {
    "batch_size": tune.choice([32, 64, 128]),
    "lr": tune.choice([0.1, 0.01, 0.001, 0.0001, 0.00001]),
    "optim": tune.choice([0, 1, 2]),
    "momentum": tune.choice([0, 0.99, 0.90]),
    "weight_decay": tune.choice([0, 0.01, 0.05, 0.1]),
    }

scheduler = ASHAScheduler(
        max_t=10,
        grace_period=1,
        reduction_factor=2)

tuner = tune.Tuner(
            tune.with_resources(
                    tune.with_parameters(trainable),
                    resources={"gpu": 1}
                ),
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                scheduler=scheduler,
                num_samples=50,
            ),
            param_space=config)

result = tuner.fit()