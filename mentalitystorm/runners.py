from abc import ABC

import torch
from tqdm import tqdm

from mentalitystorm import Storeable, MSELoss, Config, TensorBoard, OpenCV, ElasticSearchUpdater


class Runner(ABC):
    def __run__(self, config, model, dataset, batch_size, lossfunc, optimizer, epochs=2):

        device = config.device()
        if isinstance(model, Storeable):
            run_name = config.run_id_string(model)
            model.metadata['run_name'] = run_name
            model.metadata['run_url'] = config.run_url_link(model)
            model.metadata['git_commit_hash'] = config.GIT_COMMIT
            model.metadata['dataset'] = str(dataset.root)

        for epoch in tqdm(range(epochs)):
            model.train_model(dataset, batch_size=batch_size, device=device, lossfunc=lossfunc, optimizer=optimizer)
            losses = model.test_model(dataset, batch_size=batch_size, device=device, lossfunc=lossfunc)

            l = torch.Tensor(losses)

            ave_test_loss = l.mean().item()
            import math
            if not math.isnan(ave_test_loss):
                model.metadata['ave_test_loss'] = ave_test_loss

            if 'epoch' not in model.metadata:
                model.metadata['epoch'] = 1
            else:
                model.metadata['epoch'] += 1
            model.save(data_dir=config.DATA_PATH)


class ModelFactoryRunner(Runner):
    def __init__(self, model_type):
        self.model_type = model_type
        self.model_args = []
        self.model_args_index = 0
        self.optimizer_type = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.model_args_index < len(self.model_args):
            model = self.model_type(*self.model_args[self.model_args_index])
            optim =  torch.optim.Adam(model.parameters(), lr=1e-3)
            self.model_args_index += 1
            return model, optim
        else:
            raise StopIteration()

    def run(self, config, dataset, batch_size, lossfunc=MSELoss, epochs=2):
        for model, optimizer in self:
            self.__run__(config, model, dataset, batch_size, lossfunc, optimizer, epochs)


class OneShotRunner(Runner):
    def run(self, config, model, dataset, batch_size, lossfunc, optimizer, epochs=2):
        self.__run__(config, model, dataset, batch_size, lossfunc, optimizer, epochs)


""" Runner with good defaults
Uses batch size 16, runs for 10 epochs, using MSELoss and Adam Optimizer with lr 0.001
"""


class OneShotEasyRunner(Runner):
    def run(self, model, dataset, batch_size=16, epochs=10, lossfunc=None):
        config = Config()
        config.increment('run_id')

        tb = TensorBoard(config.tb_run_dir(model))
        tb.register(model)
        model.registerView('input', OpenCV('input', (160, 210)))
        model.registerView('output', OpenCV('output', (160, 210)))
        ElasticSearchUpdater().register(model)

        if lossfunc is None:
            lossfunc = MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.__run__(config, model, dataset, batch_size, lossfunc, optimizer, epochs)