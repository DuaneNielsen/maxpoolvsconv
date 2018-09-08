from mentalitystorm import Config, Storeable, ModelDb, NumpyRGBWrapper
from pathlib import Path
import torchvision
import imageio

if __name__ == '__main__':

    config = Config()

    datadir = Path(config.DATA_PATH) / 'spaceinvaders/images/raw'
    dataset = torchvision.datasets.ImageFolder(
        root=datadir.absolute(),
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    )

    test_image = dataset.__getitem__(0)
    test_image = test_image[0].unsqueeze(0)

    imageio.imwrite('test_image.jpg', NumpyRGBWrapper(test_image[0], 'tensorPIL').getImage())


    mdb = ModelDb(config.DATA_PATH)

    best_maxpooling = mdb.best_loss_for_model_class('MaxPooling')
    max_pool = Storeable.load(best_maxpooling, config.DATA_PATH)
    maxpool_image = max_pool(test_image, noise=False)

    imageio.imwrite('maxpool.jpg', NumpyRGBWrapper(maxpool_image[0].data, 'tensorPIL').getImage())
    imageio.imwrite('maxpoolz.jpg', NumpyRGBWrapper(maxpool_image[1].data, 'tensorPIL').getImage())

    best_convpooling = mdb.best_loss_for_model_class('ConvolutionalPooling')
    conv_pool = Storeable.load(best_convpooling, config.DATA_PATH)
    convpool_image = conv_pool(test_image, noise=False)

    imageio.imwrite('convpool.jpg', NumpyRGBWrapper(convpool_image[0].data, 'tensorPIL').getImage())
    imageio.imwrite('convpoolz.jpg', NumpyRGBWrapper(convpool_image[1].data, 'tensorPIL').getImage())
