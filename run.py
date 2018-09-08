from mentalitystorm import OneShotEasyRunner, Config
from pathlib import Path
import torchvision
import torchvision.transforms as TVT
import models

if __name__ == '__main__':

    datadir = Path(Config().DATA_PATH) / 'spaceinvaders/images/raw'
    dataset = torchvision.datasets.ImageFolder(
        root=datadir.absolute(),
        transform=TVT.Compose([TVT.ToTensor()])
    )

    convolutions = models.ConvolutionalPooling()

    easy = OneShotEasyRunner()
    easy.run(convolutions, dataset, batch_size=20, epochs=20)

    maxpool = models.MaxPooling()

    easy = OneShotEasyRunner()
    easy.run(maxpool, dataset, batch_size=20, epochs=20)

