import torch

class BaseTrainer():

    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
    
    def fit(self, train_loader, test_loader, epochs, device):
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.train_loop(train_loader, self.model, self.loss_fn, self.optimizer, device)
            self.test_loop(test_loader, self.model, self.loss_fn, device)
        print("Done!")

    def train_loop(self, dataloader, model, loss_fn, optimizer, device):
        # make sure the model is in train mode
        model.train()

        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)

            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 1024 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def test_loop(self, dataloader, model, loss_fn, device):
        # make sure the model is in eval mode
        model.eval()
        
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")