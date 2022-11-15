from copy import deepcopy
import os
import datetime
import json
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

    def pop_back(self):
        self.losses.pop()
        self.accuracies.pop()

class Exporter():
    def __init__(self) -> None:
        pass

    def prepare_export(self, folder, model_name):
        self.base_folder = folder
        self.date_now = datetime.datetime.now()
        self.folder = os.path.join(folder, self.date_now.strftime("%Y%m%d_%H%M") + '_' + model_name)

        os.makedirs(self.folder, exist_ok=True)
    
    def export_stat_file(self, stat_dict):
        stat_file_name = os.path.join(self.folder, 'stats.json')
        stats_file_data_json = json.dumps(stat_dict, indent=4)
        
        with open(os.path.normpath(stat_file_name), 'w', encoding = 'utf-8') as file:
            file.write(stats_file_data_json)
    
    def export_model(self, model, name):
        torch.save(model.state_dict(), os.path.join(self.folder, name))

    def export_best_models(self, models_dict):
        for k in models_dict:
            self.export_model(models_dict[k], "best_" + str(k) + "_weights.pt")

