from copy import deepcopy
from pure_eval import Evaluator
import torch
import os
from tqdm import tqdm

class ModelStats():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.best_model = None
        self.best_epoch = None
        self.best_loss = None
        self.best_accuracy = 0

        self.losses = []
        self.accuracies = []
    
    def add(self, epoch, model, loss, accuracy):
        self.losses.append(loss)
        self.accuracies.append(accuracy)

        if(accuracy > self.best_accuracy):
            self.best_model = deepcopy(model)
            self.best_epoch = epoch
            self.best_loss = loss
            self.best_accuracy = accuracy
    
    def get_stats(self):
        return {
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "best_accuracy": self.best_accuracy,
            "losses": self.losses,
            "accuracies": self.accuracies
        }
    
    def get_best_model(self):
        return self.best_model
class BaseTrainer():

    def __init__(self, model, loss_fn, optimizer, checkpoints_path=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.checkpoints_path = os.path.join(checkpoints_path, 'checkpoint')

        self.stats = {}
        self.stats['train'] = ModelStats()
        self.stats['valid'] = ModelStats()
        self.stats['test'] = ModelStats()

        self.train_size = 0
        self.train_num_batch = 0
        self.valid_size = 0
        self.valid_num_batch = 0
        self.test_size = 0
        self.test_num_batch = 0

        self.current_epoch = 0
        self.max_epoch = None
        self.device = None

        self._reset_stats()

        if checkpoints_path == None:
            self.save_checkpoint = False
        else:
            self.save_checkpoint = True
            os.makedirs(self.checkpoints_path, exist_ok=True)
        
    def fit(self, train_loader, valid_loader, test_loader, epochs, device):
        self._reset_stats()
        self.model.train()
        torch.backends.cudnn.benchmark = True
        self.current_epoch = 0
        self.max_epoch = epochs
        self.device = device

        # can't rely on dataset size when using a subset sampler
        self._count_data_from_all_loaders_and_load_to_device(train_loader, valid_loader, test_loader)
        print("Size of train dataset={0}, train batches={1}, valid dataset={2}, valid batches={3}, test dataset={4}, test batches={5}".format(self.train_size, self.train_num_batch, self.valid_size, self.valid_num_batch, self.test_size, self.test_num_batch))

        for t in range(epochs):
            self._train_loop(train_loader, device)
            
            with torch.no_grad():
                self._valid_loop(valid_loader, device)
                self._test_loop(test_loader, device)

            if self.save_checkpoint:
                torch.save({
                    'epoch': t,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.stats['train'].losses[-1],
                    'accuracy' : self.stats['train'].accuracies[-1],
                    }, os.path.join(self.checkpoints_path, 'checkpoint_' + str(t) + '.pt'))
            
            self._print_evaluation(valid_loader, test_loader)

        torch.backends.cudnn.benchmark = False

    def _print_evaluation(self, valid_loader, test_loader):
        message = "epoch {0} of {1} : train_loss: {2:.5f}, train_accuracy: {3:.2%}".format(self.current_epoch + 1, self.max_epoch, self.stats['train'].losses[-1], self.stats['train'].accuracies[-1])

        if valid_loader != None:
            message += ", valid_loss: {0:.5f}, valid_accuracy: {1:.2%}".format(self.stats['valid'].losses[-1], self.stats['valid'].accuracies[-1])
        
        if test_loader != None:
            message += ", test_loss: {0:.5f}, test_accuracy: {1:.2%}".format(self.stats['test'].losses[-1], self.stats['test'].accuracies[-1])
        
        print(message)

    def _count_data_from_loader_and_load_to_device(self, loader):
        n = 0
        nbatch = 0
        for X, y in loader:
            n += X.size(dim=0)
            nbatch += 1
            X.to(self.device)
            y.to(self.device)
        return (n, nbatch)

    def _count_data_from_all_loaders_and_load_to_device(self, train_loader, valid_loader, test_loader):
        if train_loader != None:
            (self.train_size, self.train_num_batch) = self._count_data_from_loader_and_load_to_device(train_loader)
        
        if valid_loader != None:
            (self.valid_size, self.valid_num_batch) = self._count_data_from_loader_and_load_to_device(valid_loader)
        
        if test_loader != None:
            (self.test_size, self.test_num_batch) = self._count_data_from_loader_and_load_to_device(test_loader)

    def _train_loop(self, dataloader, device):
        size = self.train_size
        num_batches = self.train_num_batch
        train_loss, accuracy = 0, 0

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
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

        train_loss /= num_batches
        accuracy /= size

        self.stats['train'].add(self.current_epoch, self.model, train_loss, accuracy)

    def _evaluate_model(self, dataloader, size, num_batches, device):
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


    def _valid_loop(self, dataloader, device):
        if dataloader == None:
            return

        loss, accuracy = self._evaluate_model(dataloader, self.valid_size, self.valid_num_batch, device)

        self.stats['valid'].add(self.current_epoch, self.model, loss, accuracy)
    
    def _test_loop(self, dataloader, device):
        if dataloader == None:
            return

        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        loss, accuracy = self._evaluate_model(dataloader, size, num_batches, device)

        self.stats['test'].add(self.current_epoch, self.model, loss, accuracy)
    
    def _reset_stats(self):
        for stat in self.stats.values():
            stat.reset()

    def get_stats(self):
        obj = {}
        for k, v in self.stats:
            obj[k] = v.get_stats()
        return obj

    def get_best_models(self):
        obj = {}
        for k, v in self.stats:
            obj[k] = v.get_best_model()
        return obj