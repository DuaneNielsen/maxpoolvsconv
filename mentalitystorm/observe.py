from abc import ABC, abstractmethod
from mentalitystorm.image import NumpyRGBWrapper
import cv2
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import imageio
from PIL import Image

"""Dispatcher allows dipatch to views.
View's register here
To send a message, inherit Observable and use updateObservers
"""


class Dispatcher:
    def __init__(self):
        self.pipelineView = {}

    def registerView(self, tag, observer):
        if tag not in self.pipelineView:
            self.pipelineView[tag] = []
        self.pipelineView[tag].append(observer)

        return tag,len(self.pipelineView[tag]) -1

    def unregisterView(self, id):
        del self.pipelineView[id[0]][id[1]]


""" Observable provides dispatch method.
To use, make sure the object has a Dispatcher 
"""


class Observable:

    def updateObserversWithImage(self, tag, image, format=None, training=True):
        metadata = {}
        metadata['func'] = 'image'
        metadata['name'] = tag
        metadata['format'] = format
        metadata['training'] = training
        self.updateObservers(tag, image, metadata)

    def updateObservers(self, tag, data, metadata=None):
        if hasattr(self, 'pipelineView'):
            if tag not in self.pipelineView:
                self.pipelineView[tag] = []
            for observer in self.pipelineView[tag]:
                observer.update(data, metadata)

    """ Sends a close event to all observers.
    used to close video files or save at the end of rollouts
    """
    def endObserverSession(self):
        if hasattr(self, 'pipelineView'):
            for tag in self.pipelineView:
                for observer in self.pipelineView[tag]:
                    observer.endSession()


""" Abstract base class for implementing View.
"""


class View(ABC):
    @abstractmethod
    def update(self, data, metadata):
        raise NotImplementedError

    def endSession(self):
        pass


class ImageVideoWriter(View):
    def __init__(self, directory, prefix):
        self.directory = directory
        self.prefix = prefix
        self.number = 0
        self.writer = None

    def update(self, screen, metadata=None):

        in_format = metadata['format'] if metadata is not None and 'format' in metadata else None

        if not self.writer:
            self.number += 1
            file = self.directory + self.prefix + str(self.number) + '.mp4'
            self.writer = imageio.get_writer(file, macro_block_size=None)

        frame = NumpyRGBWrapper(screen, in_format).numpyRGB
        self.writer.append_data(frame)

    def endSession(self):
        self.writer.close()
        self.writer = None


class ImageFileWriter(View):
    def __init__(self, directory, prefix, num_images=8192):
        super(ImageFileWriter, self).__init__()
        self.writer = None
        self.directory = directory
        self.prefix = prefix
        self.num_images = num_images
        self.imagenumber = 0

    def update(self, screen, metadata=None):

        in_format = metadata['format'] if metadata is not None and 'format' in metadata else None

        frame = NumpyRGBWrapper(screen, in_format).numpyRGB
        Image.fromarray(frame).save(self.directory + '/' + self.prefix + str(self.imagenumber) + '.png')
        self.imagenumber = (self.imagenumber + 1) % self.num_images



class OpenCV(View):
    def __init__(self, title='title', screen_resolution=(640,480)):
        super(OpenCV, self).__init__()
        self.C = None
        self.title = title
        self.screen_resolution = screen_resolution

    def update(self, screen, metadata=None):

        format = metadata['format'] if metadata is not None and 'format' in metadata else None

        frame = NumpyRGBWrapper(screen, format)
        frame = frame.getImage()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, self.screen_resolution)

        # Display the resulting frame
        cv2.imshow(self.title, frame)
        cv2.waitKey(1)


class Plotter():

    def __init__(self, figure):
        self.image = None
        self.figure = figure
        plt.ion()

    def setInput(self, input):
        if input == 'numpyRGB':
            self.C = lambda x : x

    def update(self, screen, metadata=None):

        plt.figure(self.figure)

        image = self.C(screen)

        if self.image is None:
            self.image = plt.imshow(image)
        else:
            self.image.set_array(image)

        plt.pause(0.001)
        #plt.draw()


class TensorBoard(View, SummaryWriter):
    def __init__(self, run=None, comment='default', image_freq=50):
        View.__init__(self)
        SummaryWriter.__init__(self, run, comment)
        self.image_freq = image_freq
        self.dispatch = {'tb_step': self.step, 'tb_scalar':self.scalar, 'image':self.image}
        self.global_step = None

    def register(self, model):
        model.registerView('tb_step', self)
        model.registerView('tb_training_loss', self)
        model.registerView('tb_test_loss', self)
        model.registerView('input', self)
        model.registerView('output', self)
        model.registerView('z', self)
        model.registerView('tb_train_time', self)
        model.registerView('tb_train_time_per_item', self)


    def update(self, data, metadata):
        func = self.dispatch.get(metadata['func'])
        func(data, metadata)


    def step(self, data, metadata):
        self.global_step = metadata['tb_global_step']

    def scalar(self, value, metadata):
        if self.global_step:
            self.add_scalar(metadata['name'], value, self.global_step)

    def image(self, value, metadata):
        if self.global_step and self.global_step % self.image_freq == 0 and not metadata['training']:
            self.add_image(metadata['name'], value, self.global_step)

""" Convenience methods for dispatch to tensorboard
requires that the object also inherit Observable
"""


# noinspection PyUnresolvedReferences
class TensorBoardObservable:
    def __init__(self):
        self.global_step = 0
        if hasattr(self, 'metadata') and 'tb_global_step' in self.metadata:
            self.global_step = self.metadata['tb_global_step']

    def tb_global_step(self):
        self.global_step += 1
        self.updateObservers('tb_step', None, {'func': 'tb_step', 'tb_global_step': self.global_step})
        if hasattr(self, 'metadata') and 'tb_global_step' in self.metadata:
            self.metadata['tb_global_step'] = self.global_step

    def writeScalarToTB(self, tag, value, tb_name):
        self.updateObservers(tag, value,
                             {'func': 'tb_scalar',
                              'name': tb_name})

    def writeTrainingLossToTB(self, loss):
        self.writeScalarToTB('tb_training_loss', loss, 'loss/train')

    def writeTestLossToTB(self, loss):
        self.writeScalarToTB('tb_test_loss', loss, 'loss/test')

    def writePerformanceToTB(self, time, batch_size):
        self.writeScalarToTB('tb_train_time', time, 'perf/train_time_per_batch')
        if batch_size != 0:
            self.writeScalarToTB('tb_train_time_per_item', time/batch_size, 'perf/train_time_per_item')



class SummaryWriterWithGlobal(SummaryWriter):
    def __init__(self, comment):
        super(SummaryWriterWithGlobal, self).__init__(comment=comment)
        self.global_step = 0

    def tensorboard_step(self):
        self.global_step += 1

    def tensorboard_scaler(self, name, scalar):
        self.add_scalar(name, scalar, self.global_step)

    """
    Adds a matplotlib plot to tensorboard
    """
    def plotImage(self, plot):
        self.add_image('Image', plot.getPlotAsTensor(), self.global_step)
        plot.close()
