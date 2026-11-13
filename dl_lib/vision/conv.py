import torch

from dl_lib.nn.module import Module


class Conv1d(Module):
    def __init__(
        self, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0
    ):
        super().__init__()
        self.weight = torch.randn(
            out_channels, in_channels, kernel_size, requires_grad=True
        )
        self.bias = torch.rand(out_channels, requires_grad=True)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def _init_result(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        expected_length = (
            length + 2 * self.padding - self.kernel_size
        ) // self.stride + 1

        batch_size = x.shape[0]
        res = torch.ones(batch_size, self.out_channels, expected_length)
        return res

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        # Handle padding
        if self.padding > 0:
            batch_size = x.shape[0]
            new_x = torch.zeros(batch_size, self.in_channels, length + self.padding * 2)
            new_x[:, :, self.padding : self.padding + length] = x
            return new_x
        return x

    def forward(self, x):
        # (in_channels, length) or (batch, in_channels, length)
        # Handle output
        unbatched = x.ndim == 2
        if unbatched:
            x = x.unsqueeze(0)
        res = self._init_result(x)

        # Handle padding
        x = self._pad_input(x)

        for current_index in range(res.shape[-1]):
            i = current_index * self.stride

            batch_size = x.shape[0]
            # x: batch_size , input_chanel , kernel_size -> batch_size , 1 , input_chanel , kernel_size
            # w: out_channels , in_channels , kernel_size  -> 1 , out_channels , in_channels , kernel_size
            # Should broadcast
            res[:, :, current_index] = (
                x[:, :, i : i + self.kernel_size].view(
                    batch_size, 1, self.in_channels, self.kernel_size
                )
                * self.weight.view(
                    1, self.out_channels, self.in_channels, self.kernel_size
                )
            ).sum(dim=(-1, -2)) + self.bias
        return res if not unbatched else res.squeeze(0)

    def parameters(self):
        return [self.weight, self.bias]


class Conv2d(Module):
    def __init__(
        self, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0
    ):
        super().__init__()

        self.weight = torch.randn(
            out_channels,
            in_channels,
            kernel_size,
            kernel_size,
            requires_grad=True,
        )
        self.bias = torch.rand(out_channels, requires_grad=True)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def _init_result(self, x: torch.Tensor) -> torch.Tensor:
        width, height = x.shape[-2], x.shape[-1]

        expected_height = (
            height + 2 * self.padding - self.kernel_size
        ) // self.stride + 1
        expected_width = (
            width + 2 * self.padding - self.kernel_size
        ) // self.stride + 1

        batch_size = x.shape[0]
        res = torch.ones(batch_size, self.out_channels, expected_width, expected_height)
        return res

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        width, height = x.shape[-2], x.shape[-1]
        # Handle padding
        if self.padding > 0:
            batch_size = x.shape[0]
            new_x = torch.zeros(
                batch_size,
                self.in_channels,
                width + self.padding * 2,
                height + self.padding * 2,
            )
            new_x[
                :,
                :,
                self.padding : self.padding + width,
                self.padding : self.padding + height,
            ] = x
            return new_x
        return x

    def forward(self, x):
        # (in_channels, width, height) or (batch, in_channels, width, height)
        # Handle output
        unbatched = x.ndim == 3
        if unbatched:
            x = x.unsqueeze(0)
        res = self._init_result(x)

        # Handle padding
        x = self._pad_input(x)
        for current_row in range(res.shape[-2]):
            for current_col in range(res.shape[-1]):
                row_index = current_row * self.stride
                col_index = current_col * self.stride

                batch_size = x.shape[0]
                # x: batch_size , input_chanel , kernel_size -> batch_size , 1 , input_chanel , kernel_size
                # w: out_channels , in_channels , kernel_size  -> 1 , out_channels , in_channels , kernel_size
                # Should broadcast
                res[:, :, current_row, current_col] = (
                    x[
                        :,
                        :,
                        row_index : row_index + self.kernel_size,
                        col_index : col_index + self.kernel_size,
                    ].view(
                        batch_size,
                        1,
                        self.in_channels,
                        self.kernel_size,
                        self.kernel_size,
                    )
                    * self.weight.view(
                        1,
                        self.out_channels,
                        self.in_channels,
                        self.kernel_size,
                        self.kernel_size,
                    )
                ).sum(dim=(-1, -2, -3)) + self.bias
        return res if not unbatched else res.squeeze(0)

    def parameters(self):
        return [self.weight, self.bias]
