from copy import deepcopy
import torch
import os
from tqdm import tqdm

class BaseTrainer():

    def __init__(self, model, loss_fn, optimizer, checkpoints_path=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.checkpoints_path = os.path.join(checkpoints_path, 'checkpoint')

        self.best_model = None
        self.best_epoch = None
        self.best_loss = None
        self.best_accuracy = 0

        self.reset_stats()

        if checkpoints_path == None:
            self.save_checkpoint = False
        else:
            self.save_checkpoint = True
            os.makedirs(self.checkpoints_path, exist_ok=True)
        
    def fit(self, train_loader, test_loader, epochs, device):
        self.reset_stats()
        self.model.train()

        for t in tqdm(range(epochs)):

            self.train_loop(train_loader, device)
            self.test_loop(test_loader, device)
            

            if self.stats['test_accuracy'][-1] > self.best_accuracy:
                self.best_accuracy = self.stats['test_accuracy'][-1]
                self.best_epoch = t
                self.best_loss = self.stats['test_loss'][-1]
                self.best_model = deepcopy(self.model.state_dict())

            if self.save_checkpoint:
                torch.save({
                    'epoch': t,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.stats['train_loss'][-1],
                    'accuracy' : self.stats['train_accuracy'][-1],
                    }, os.path.join(self.checkpoints_path, 'checkpoint_' + str(t) + '.pt'))

    def train_loop(self, dataloader, device):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        train_loss, correct = 0, 0

        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)

            # Compute prediction and loss
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        train_loss /= num_batches
        correct /= size

        self.stats['train_loss'].append(train_loss)
        self.stats['train_accuracy'].append(correct)

    def test_loop(self, dataloader, device):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size

        self.stats['test_loss'].append(test_loss)
        self.stats['test_accuracy'].append(correct)
    
    def reset_stats(self):
        self.stats = {
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": []
            }

    def get_stats(self):
        return self.stats

    def get_best_model(self):
        return (self.best_epoch, self.best_accuracy, self.best_loss, self.best_model)