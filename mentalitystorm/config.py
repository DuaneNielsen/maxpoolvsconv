import os
from urllib.parse import quote
import torch
import torchvision
from torchvision import transforms as TVT
from pathlib import Path
import json
import logging
from logging.handlers import TimedRotatingFileHandler

class Config:
    def __init__(self):
        # environment variables
        self.BUILD_TAG = os.environ.get('BUILD_TAG', 'build_tag').replace('"', '')
        self.GIT_COMMIT = os.environ.get('GIT_COMMIT', 'git_commit').replace('"', '')
        self.DATA_PATH = os.environ.get('DATA_PATH', 'c:\data').replace('"', '')
        self.TORCH_DEVICE = os.environ.get('TORCH_DEVICE', 'cuda').replace('"', '')

        self.configpath = Path(self.DATA_PATH) / 'config.json'
        if self.configpath.exists():
            self.config = Config.load(self.configpath.absolute())
        else:
            self.config = {}
            self.config['run_id'] = 0
            self.save(self.configpath.absolute())

        logfile = self.getLogPath('most_improved.log')
        logging.basicConfig(filename=logfile.absolute())

    def rolling_run_number(self):
        return "{0:0=3d}".format(self.config['run_id']%1000)

    def run_id_string(self, model):
        return 'runs/run' + self.rolling_run_number() + '/' +  model.metadata['slug']

    def convert_to_url(self, run, host=None, port='6006'):
        if host is None:
            import socket
            host = socket.gethostname()
        url = run.replace('\\', '\\\\')
        url = run.replace('/', '\\\\')
        url = quote(url)
        url = 'http://' + host + ':' + port + '/#scalars&regexInput=' + url
        return url

    def run_url_link(self, model):
        run = self.run_id_string(model)
        url = self.convert_to_url(run)
        return url

    def tb_run_dir(self, model):
        return self.DATA_PATH + '/' + self.run_id_string(model)

    def device(self):
        return torch.device(str(self.TORCH_DEVICE))

    def __str__(self):
        return 'DATA_PATH ' +  str(self.DATA_PATH) + \
               ' GIT_COMMIT ' + str(self.GIT_COMMIT) + \
               ' TORCH_DEVICE ' + str(self.TORCH_DEVICE)

    def dataset(self, datapath):
        datadir = Path(self.DATA_PATH).joinpath(datapath)
        dataset = torchvision.datasets.ImageFolder(
            root=datadir.absolute(),
            transform=TVT.Compose([TVT.ToTensor()])
        )
        return dataset

    def getLogPath(self, name):
        logfile = Path(self.DATA_PATH) / 'logs' / name
        logfile.parent.mkdir(parents=True, exist_ok=True)
        return logfile

    def update(self, key, value):
        self.config[key] = value
        self.save(self.configpath)

    def increment(self, key):
        self.config[key] += 1
        self.save(self.configpath)

    def save(self, filename):
        with open(filename, 'w') as configfile:
            json.dump(self.config, fp=configfile, indent=2)

    @staticmethod
    def load(filename):
        with open(filename, 'r') as configfile:
            return json.load(fp=configfile)
