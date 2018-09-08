import torch
import torchvision.transforms as TVT
import numpy as np

tensorPILTonumpyRBG = lambda tensor : tensor.squeeze().permute(1, 2, 0).cpu().numpy()
tensorGreyscaleTonumpyRGB = lambda tensor : tensor.expand(3,-1,-1).squeeze().permute(1, 2, 0).cpu().numpy()

class BaseImageWrapper():
    def __init__(self, image, format=None):
        self.image = image
        self.format = format
        if format is None:
            self.format = self.guess_format(image)

    def guess_format(self, image):
            # guess it based on the screen
            if type(image) == torch.Tensor:
                if image.shape[0] == 3:
                    return 'tensorPIL'
                elif image.shape[0] == 1:
                    return 'tensorGreyscale'
            elif type(image) == np.ndarray:
                if image.shape[0] == 3:
                    return 'numpyRGB'
                elif image.shape[0] == 1:
                    return 'numpyGreyscale'
                elif image.shape[2] == 3:
                    return 'numpyRGB3'
            else:
                raise Exception('failed to autodetect format please specify format')

class NumpyRGBWrapper(BaseImageWrapper):
    def __init__(self, image, format=None):
        super(NumpyRGBWrapper, self).__init__(image, format)
        self.numpyRGB = None
        if self.format == 'numpyRGB':
            self.numpyRGB = self.image
        elif self.format == 'tensorPIL':
            self.numpyRGB =  tensorPILTonumpyRBG(self.image)
        elif self.format == 'tensorGreyscale':
            TF = TVT.Compose([TVT.ToPILImage(),TVT.Grayscale(3),TVT.ToTensor()])
            tensor_PIL = TF(image.cpu())
            self.numpyRGB = tensorPILTonumpyRBG(tensor_PIL)
        elif self.format == 'numpyGreyscale':
            self.numpyRGB = np.repeat(image, 3, axis=0)
        elif self.format == 'numpyRGB3':
            self.numpyRGB = np.transpose(image, [2,0,1])
        else:
            raise Exception('conversion ' + self.format + ' to numpyRGB not implemented')

    def getImage(self):
        return self.numpyRGB

class TensorPILWrapper(BaseImageWrapper):
    def __init__(self, image, format=None):
        BaseImageWrapper.__init__(self, image, format)
        self.tensorPIL = None
        if self.format == 'tensorPIL':
            self.tensorPIL = self.image
        elif self.format == 'numpyRGB':
            # I don't think this works..
            self.tensorPIL =  tensorPILTonumpyRBG(self.image)
        elif self.format == 'numpyRGB3':
            frame = image.transpose(2, 0, 1)
            frame = np.flip(frame, axis=0)
            frame = np.copy(frame)
            TF = TVT.Compose([TVT.ToPILImage(),TVT.ToTensor])
            self.tensorPIL = torch.from_numpy(frame)
        else:
            raise Exception('conversion ' + str(self.format) + ' to tensorPIL not implemented')

    def getImage(self):
        return self.tensorPIL