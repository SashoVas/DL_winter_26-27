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
        # (in_channels, length) or (batch, in_channels, length)

        length = x.shape[-1]
        if x.ndim == 2:
            res = torch.ones(self.out_channels, length-self.kernel_size+1)
        else:
            batch_size = x.shape[0]
            res = torch.ones(
                batch_size, self.out_channels, length-self.kernel_size+1)

        for i in range(0, length-self.kernel_size+1):
            if x.ndim == 2:
                # x: input_chanel , kernel_size
                # w: out_channels, in_channels, kernel_size
                # Should broadcast
                res[:, i] = (x[:,
                               i: i+self.kernel_size]*self.weight).sum(dim=(1, 2)) + self.bias
            else:
                # x: batch_size , input_chanel , kernel_size -> batch_size , 1 , input_chanel , kernel_size
                # w: out_channels , in_channels , kernel_size  -> 1 , out_channels , in_channels , kernel_size
                res[:, :, i] = (
                    x[:, :, i: i+self.kernel_size].view(
                        batch_size, 1, self.in_channels, self.kernel_size)
                    * self.weight.view(1, self.out_channels, self.in_channels, self.kernel_size)).sum(dim=(-1, -2)) + self.bias
        return res
