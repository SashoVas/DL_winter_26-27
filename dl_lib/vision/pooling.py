import torch

from dl_lib.nn.module import Module


class MaxPool2d(Module):
    def __init__(
        self, kernel_size: int = 3, stride: None | int = None, padding: int = 0
    ):
        super().__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def _init_result(self, x: torch.Tensor) -> torch.Tensor:
        width, height, in_channels = x.shape[-2], x.shape[-1], x.shape[-3]

        expected_height = (
            height + 2 * self.padding - self.kernel_size
        ) // self.stride + 1
        expected_width = (
            width + 2 * self.padding - self.kernel_size
        ) // self.stride + 1

        batch_size = x.shape[0]
        res = torch.ones(batch_size, in_channels, expected_width, expected_height)
        return res

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        width, height, in_channels = x.shape[-2], x.shape[-1], x.shape[-3]
        # Handle padding
        if self.padding > 0:
            batch_size = x.shape[0]
            new_x = torch.full(
                (
                    batch_size,
                    in_channels,
                    width + self.padding * 2,
                    height + self.padding * 2,
                ),
                float("-inf"),
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
                res[:, :, current_row, current_col] = x[
                    :,
                    :,
                    row_index : row_index + self.kernel_size,
                    col_index : col_index + self.kernel_size,
                ].amax(dim=(-2, -1))

        return res if not unbatched else res.squeeze(0)
