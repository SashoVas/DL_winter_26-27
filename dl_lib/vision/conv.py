from dl_lib.nn.module import Module
import torch


class Conv1d(Module):
    def __init__(self,  in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.weight = torch.randn(
            out_channels, in_channels, kernel_size, requires_grad=True)
        self.bias = torch.rand(out_channels, requires_grad=True)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        res = torch.ones(self.out_channels, x.shape[1]-self.kernel_size+1)
        for i in range(0, x.shape[1]-self.kernel_size+1):
            res[:, i: i+self.kernel_size] = (x[:,
                                               i: i+self.kernel_size]*self.weight).sum(dim=(1, 2)) + self.bias
        return res
