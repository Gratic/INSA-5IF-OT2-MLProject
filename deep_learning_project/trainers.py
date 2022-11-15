from pure_eval import Evaluator
import torch
import os
from .utils import ModelStats
from ray import tune
from ray.air import session, Checkpoint
from tqdm import tqdm

class BaseTrainer():

    def __init__(self, model, loss_fn, optimizer, checkpoints_path=None, tunning=False):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

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
        self.max_epochs = None
        self.device = None

        self.tunning = tunning

        self._reset_stats()

        if checkpoints_path == None:
            self.save_checkpoint = False
        else:
            self.save_checkpoint = True
            self.checkpoints_path = os.path.join(checkpoints_path, 'checkpoints')
            os.makedirs(self.checkpoints_path, exist_ok=True)
        
    def fit(self, train_loader, valid_loader=None, test_loader=None, min_epochs=None, max_epochs=10, early_stopping=None, device='cpu', verbose=True):
        self.model.train()
        torch.backends.cudnn.benchmark = True
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.early_stopping = early_stopping
        self.device = device
        
        last_checkpoint = None

        assert early_stopping in [None, 'train', 'test', 'valid'], "Early stopping must evaluate to None, train, test, or valid."
        
        if early_stopping == 'test':
            assert test_loader != None, "Early stopping evaluate to test, but there is no test_loader."
        
        if early_stopping == 'valid':
            assert valid_loader != None, "Early stopping evaluate to valid, but there is no valid_loader."


        # can't rely on dataset size when using a subset sampler
        self._count_data_from_all_loaders_and_load_to_device(train_loader, valid_loader, test_loader)
        print("Size of train dataset={0}, train batches={1}, valid dataset={2}, valid batches={3}, test dataset={4}, test batches={5}".format(self.train_size, self.train_num_batch, self.valid_size, self.valid_num_batch, self.test_size, self.test_num_batch))

        if self.tunning and session.get_checkpoint():
            loaded_checkpoint = session.get_checkpoint()
            with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
                path = os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
                checkpoint = torch.load(path)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.current_epoch = checkpoint["epoch"]

        for t in range(max_epochs):
            self._train_loop(train_loader, device)
            
            with torch.no_grad():
                self._valid_loop(valid_loader, device)
                self._test_loop(test_loader, device)
            
            checkpoint = {
                'epoch': t,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.stats['train'].losses[-1],
                'accuracy' : self.stats['train'].accuracies[-1],
                }

            if self.tunning:
                # torch.save(checkpoint, os.path.join(self.checkpoints_path, 'checkpoint.pt'))
                session.report({"loss":self.stats['valid'].losses[-1], "accuracy":self.stats['valid'].accuracies[-1]})
                # session.report({"loss":self.stats['valid'].losses[-1], "accuracy":self.stats['valid'].accuracies[-1]}, checkpoint=Checkpoint.from_directory(self.checkpoints_path))

            if not self.tunning or verbose:
                self._print_evaluation(valid_loader, test_loader)

            if self.early_stopping != None and self.stats[self.early_stopping].losses[-1] > self.stats[self.early_stopping].best_loss:
                self.pop_back_stats()
                break
            elif self.save_checkpoint:
                last_checkpoint = 'checkpoint_' + str(t) + '.pt'
                torch.save(checkpoint, os.path.join(self.checkpoints_path, last_checkpoint))
            
            self.current_epoch += 1

        torch.backends.cudnn.benchmark = False

    def _print_evaluation(self, valid_loader, test_loader):
        message = "epoch {0} of {1} : train_loss: {2:.5f}, train_accuracy: {3:.2%}".format(self.current_epoch + 1, self.max_epochs, self.stats['train'].losses[-1], self.stats['train'].accuracies[-1])

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

        for batch, (X, y) in tqdm(enumerate(dataloader), disable=self.tunning):
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
        for k in self.stats:
            obj[k] = self.stats[k].get_stats()
        return obj
    
    def pop_back_stats(self):
        for k in self.stats:
            self.stats[k].pop_back()

    def get_best_models(self):
        obj = {}
        for k in self.stats:
            obj[k] = self.stats[k].get_best_model()
        return obj