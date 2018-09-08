import torch

"""
Utility function for computing output of convolutions
takes a tuple of (h,w) and returns a tuple of (h,w)
"""
def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)

    if type(pad) is not tuple:
        pad = (pad, pad)

    h = floor( ((h_w[0] + (2 * pad[0]) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad[1]) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w

def conv_transpose_output_shape(h_w, kernel_size=1, stride=1, pad=0, output_padding=0):
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = (h_w[0] - 1) * stride - (2 * pad) + kernel_size[0] + output_padding
    w = (h_w[1] - 1) * stride - (2 * pad) + kernel_size[1] + output_padding
    return h, w

""" Generates a default index map for nn.MaxUnpool2D operation.
output_shape: the shape that was put into the nn.MaxPool2D operation
in terms of nn.MaxUnpool2D this will be the output_shape
pool_size: the kernel size of the MaxPool2D
"""
def default_maxunpool_indices(output_shape, kernel_size, batch_size, channels, device):
    ph = kernel_size[0]
    pw = kernel_size[1]
    h = output_shape[0]
    w = output_shape[1]
    ih = output_shape[0] // 2
    iw = output_shape[1] // 2
    h_v = torch.arange(ih,dtype=torch.int64, device=device) * pw  * ph * iw
    w_v = torch.arange(iw,dtype=torch.int64, device=device) * pw
    h_v = torch.transpose(h_v.unsqueeze(0), 1,0)
    return (h_v + w_v).expand(batch_size, channels, -1, -1)