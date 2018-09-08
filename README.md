# MaxPooling vs Convolutions

Code for my a [article on medium](https://medium.com/@duanenielsen/deep-learning-cage-match-max-pooling-vs-convolutions-e42581387cb9)

An experiment to compare the performance of 2x2 max-pooling vs using a 2x2 convolution with a stride of 2.

Tested by auto-encoding images generated from openai/gym

The encoder networks consist of a convolution layer with 3 filters kernel_size=3, stride=1, with Relu activation,
followed by either a 2x2 max pooling layer or a 2x2 convolution.

The decoder network is a either a max unpooling operation (using indices) or a 2x2 ConvTranspose
layer followed by a Conv Transpose with 3 filters, kernel size 3 and stride 1, then Relu activation.

Dataset is 10,000 images generated by running a random policy on ai-gym space invaders.
Train/Test split is 90%/10%.

MSE Loss is used.  Optimizer is Adam with lr=0.001.

**Test Results after 20 epochs of training**

MSE loss on test set
![Alt text](images/maxvsconv.JPG?raw=true "Runs")

**Comparison**

![Alt text](images/comparison.jpg?raw=true "Comparison")

Left: test image, Center: Max Pooling, Right: Convolutional pooling

**Test images**

![Alt text](images/original.JPG?raw=true "Title")

**Reconstructed images**

![Alt text](images/maxvconvimage.JPG?raw=true "Title")