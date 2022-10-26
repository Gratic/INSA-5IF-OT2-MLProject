from copy import deepcopy
from pure_eval import Evaluator
import torch
import os
from tqdm import tqdm

class BaseTrainer():

    def __init__(self, model, loss_fn, optimizer, checkpoints_path=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.checkpoints_path = os.path.join(checkpoints_path, 'checkpoint')

        self.best_valid_model = None
        self.best_valid_epoch = None
        self.best_valid_loss = None
        self.best_valid_accuracy = 0

        self.best_test_model = None
        self.best_test_epoch = None
        self.best_test_loss = None
        self.best_test_accuracy = 0

        self.train_size = 0
        self.train_num_batch = 0
        self.valid_size = 0
        self.valid_num_batch = 0

        self.reset_stats()

        if checkpoints_path == None:
            self.save_checkpoint = False
        else:
            self.save_checkpoint = True
            os.makedirs(self.checkpoints_path, exist_ok=True)
        
    def fit(self, train_loader, valid_loader, test_loader, epochs, device):
        self.reset_stats()
        self.model.train()

        # can't rely on dataset size when using a subset sampler
        self.train_size = 0
        self.train_num_batch = 0
        for X,y in train_loader:
            self.train_size += X.size(dim=0)
            self.train_num_batch += 1
        
        self.valid_size = 0
        self.valid_num_batch = 0
        for X, y in valid_loader:
            self.valid_size += X.size(dim=0)
            self.valid_num_batch += 1

        print("Size of train dataset={0}, train batches={1}, valid dataset={2}, valid batches={3}".format(self.train_size, self.train_num_batch, self.valid_size, self.valid_num_batch))

        for t in range(epochs):

            self.train_loop(train_loader, device)
            self.valid_loop(valid_loader, device)

            if test_loader != None:
                self.test_loop(test_loader, device)
            

            if self.stats['valid_accuracy'][-1] > self.best_valid_accuracy:
                self.best_valid_accuracy = self.stats['valid_accuracy'][-1]
                self.best_valid_epoch = t
                self.best_valid_loss = self.stats['valid_loss'][-1]
                self.best_valid_model = deepcopy(self.model.state_dict())
            
            if test_loader != None:
                if self.stats['test_accuracy'][-1] > self.best_test_accuracy:
                    self.best_test_accuracy = self.stats['test_accuracy'][-1]
                    self.best_test_epoch = t
                    self.best_test_loss = self.stats['test_loss'][-1]
                    self.best_test_model = deepcopy(self.model.state_dict())

            if self.save_checkpoint:
                torch.save({
                    'epoch': t,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.stats['train_loss'][-1],
                    'accuracy' : self.stats['train_accuracy'][-1],
                    }, os.path.join(self.checkpoints_path, 'checkpoint_' + str(t) + '.pt'))
            
            if test_loader == None:
                print("epoch {0} of {1}: train_loss: {2:.5f}, train_accuracy: {3:.2%}, valid_loss: {4:.5f}, valid_accuracy: {5:.2%}"
                    .format(t+1,
                            epochs,
                            self.stats['train_loss'][-1],
                            self.stats['train_accuracy'][-1],
                            self.stats['valid_loss'][-1],
                            self.stats['valid_accuracy'][-1]))
            else:
                print("epoch {0} of {1}: train_loss: {2:.5f}, train_accuracy: {3:.2%}, valid_loss: {4:.5f}, valid_accuracy: {5:.2%}, test_loss: {6:.5f}, test_accuracy: {7:.2%}"
                    .format(t+1,
                            epochs,
                            self.stats['train_loss'][-1],
                            self.stats['train_accuracy'][-1],
                            self.stats['valid_loss'][-1],
                            self.stats['valid_accuracy'][-1],
                            self.stats['test_loss'][-1],
                            self.stats['test_accuracy'][-1]))

    def train_loop(self, dataloader, device):
        size = self.train_size
        num_batches = self.train_num_batch
        train_loss, correct = 0, 0

        for batch, (X, y) in tqdm(enumerate(dataloader)):
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

    def evaluate_model(self, dataloader, size, num_batches, device):
        loss, accuracy = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = self.model(X)
                loss += self.loss_fn(pred, y).item()
                accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

        loss /= num_batches
        accuracy /= size

        return (loss, accuracy)


    def valid_loop(self, dataloader, device):
        loss, correct = self.evaluate_model(dataloader, self.valid_size, self.valid_num_batch, device)

        self.stats['valid_loss'].append(loss)
        self.stats['valid_accuracy'].append(correct)
    
    def test_loop(self, dataloader, device):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        loss, correct = self.evaluate_model(dataloader, size, num_batches, device)

        self.stats['test_loss'].append(loss)
        self.stats['test_accuracy'].append(correct)
    
    def reset_stats(self):
        self.stats = {
            "train_loss": [],
            "train_accuracy": [],
            "valid_loss": [],
            "valid_accuracy": [],
            "test_loss": [],
            "test_accuracy": [],
            }

    def get_stats(self):
        return self.stats

    def get_best_model(self):
        return { 
                    "valid":
                    {
                        "epoch": self.best_valid_epoch,
                        "accuracy": self.best_valid_accuracy,
                        "loss": self.best_valid_loss,
                        "model": self.best_valid_model
                    },
                    "test":
                    {
                        "epoch": self.best_test_epoch,
                        "accuracy": self.best_test_accuracy,
                        "loss": self.best_test_loss,
                        "model": self.best_test_model
                    }
                }