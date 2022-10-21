import torch

class BaseTrainer():

    def __init__(self, model, loss_fn, optimizer, ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.reset_stats()
    
    def fit(self, train_loader, test_loader, epochs, device):
        self.reset_stats()
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train_loop(train_loader, device)
            self.test_loop(test_loader, device)
        print("Done!")

    def train_loop(self, dataloader, device):
        # make sure the model is in train mode
        self.model.train()

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

            if batch % 1024 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            train_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        train_loss /= num_batches
        correct /= size

        self.stats['train_loss'].append(train_loss)
        self.stats['train_accuracy'].append(correct)




    def test_loop(self, dataloader, device):
        # make sure the model is in eval mode
        self.model.eval()
        
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

        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    def reset_stats(self):
        self.stats = {
            "train_loss": [],
            "train_accuracy": [],
            "test_loss": [],
            "test_accuracy": []
            }

    def get_stats(self):
        return self.stats