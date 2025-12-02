import torch as th
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np


class Chomp1d(nn.Module):
    """
    PyTorch doesn't provide native causal 1D convolutions. We emulate them by
    using standard zero-padded convolutions and then chopping off the last
    `chomp_size` time steps so that outputs only depend on past inputs.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class FirstBlock(nn.Module):
    def __init__(self, target, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(FirstBlock, self).__init__()

        self.target = target
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=n_outputs
        )
        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.relu = nn.PReLU(n_inputs)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.conv1.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out = self.net(x)
        return self.relu(out)


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=n_outputs
        )
        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.relu = nn.PReLU(n_inputs)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.conv1.weight.data.normal_(0, 0.1)

    def forward(self, x):
        out = self.net(x)
        return self.relu(out + x)  # residual connection


class LastBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(LastBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=n_outputs
        )
        self.chomp1 = Chomp1d(padding)
        self.net = nn.Sequential(self.conv1, self.chomp1)
        self.linear = nn.Linear(n_inputs, n_inputs)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        # residual connection in (B, T, C) space
        return self.linear(out.transpose(1, 2) + x.transpose(1, 2)).transpose(1, 2)


class DepthwiseNet(nn.Module):
    def __init__(self, target, num_inputs, num_levels, kernel_size=2, dilation_c=2):
        super(DepthwiseNet, self).__init__()
        layers = []
        in_channels = num_inputs
        out_channels = num_inputs
        for l in range(num_levels):
            dilation_size = dilation_c ** l
            if l == 0:
                layers += [
                    FirstBlock(
                        target, in_channels, out_channels, kernel_size,
                        stride=1, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size
                    )
                ]
            elif l == num_levels - 1:
                layers += [
                    LastBlock(
                        in_channels, out_channels, kernel_size,
                        stride=1, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size
                    )
                ]
            else:
                layers += [
                    TemporalBlock(
                        in_channels, out_channels, kernel_size,
                        stride=1, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size
                    )
                ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ADDSTCN(nn.Module):
    # Train an individual network for each feature/target.
    def __init__(self, target, input_size, num_levels, kernel_size, cuda, dilation_c):
        super(ADDSTCN, self).__init__()

        self.target = target
        self.dwn = DepthwiseNet(self.target, input_size, num_levels, kernel_size=kernel_size, dilation_c=dilation_c)
        self.pointwise = nn.Conv1d(input_size, 1, 1)

        self._attention = th.ones(input_size, 1)
        self._attention = Variable(self._attention, requires_grad=False)

        self.fs_attention = th.nn.Parameter(self._attention.data)

        if cuda:
            self.dwn = self.dwn.cuda()
            self.pointwise = self.pointwise.cuda()
            self._attention = self._attention.cuda()

    def init_weights(self):
        self.pointwise.weight.data.normal_(0, 0.1)

    def forward(self, x):
        y1 = self.dwn(x * F.softmax(self.fs_attention, dim=0))
        y1 = self.pointwise(y1)
        return y1.transpose(1, 2)


class MultiADDSTCN(nn.Module):
    # Train a single network jointly for all features.
    """
    One-shot imputer for all features.

    Args:
        causal_mask: ndarray/torch.BoolTensor of shape (N, N) — parent graph with diag = 1
    """
    def __init__(self, causal_mask, num_levels, kernel_size=2, dilation_c=2, cuda=False):
        super().__init__()
        N = causal_mask.shape[0]
        self.register_buffer('C', th.tensor(causal_mask, dtype=th.float32))  # (N, N)

        # Learnable attention; gradients flow only through positions where C == 1
        self.A = nn.Parameter(th.ones_like(self.C))

        self.dwn = DepthwiseNet(
            target=None,            # target index not required here
            num_inputs=N,
            num_levels=num_levels,
            kernel_size=kernel_size,
            dilation_c=dilation_c,
        )
        # Column-wise (per-feature) outputs with depthwise separation preserved
        self.pointwise = nn.Conv1d(N, N, kernel_size=1, groups=N)

        if cuda:
            self.cuda()

    def forward(self, x):
        """
        x: (B, T, N) — full table values, missing already filled
        Internally processed as (B, N, T).
        """
        # (B, T, N) -> (B, N, T)
        x = x.transpose(1, 2)

        B, N, T = x.shape

        # Masked/soft attention constrained by C
        gate_logits = self.A * self.C - 1e6 * (1 - self.C)
        gate = th.softmax(gate_logits, dim=0)  # (N, N)

        gate_b = gate.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        x_b = x.unsqueeze(2)                      # (B, N, 1, T)

        x_gated = (x_b * gate_b).sum(dim=1)       # (B, N, T)

        h = self.dwn(x_gated)                     # (B, N, T)
        out = self.pointwise(h)                   # (B, N, T)

        return out.transpose(1, 2)                # (B, T, N) — match input layout


class ParallelFeatureADDSTCN(nn.Module):
    # Train an ADDSTCN per feature, but process all features in parallel using channel-wise grouping.
    """
    Train a collection of ADDSTCN models in parallel, one per column/feature.

    Input:
        x: (B, T, N) — multiple features/columns

    Output:
        (B, T, N) — predictions for all features
    """
    def __init__(self, causal_matrix, model_params, cuda=False):
        super().__init__()
        N = causal_matrix.shape[0]
        self.models = nn.ModuleList()

        for i in range(N):
            causal_mask = causal_matrix[:, i]               # take column i
            input_idx = np.where(causal_mask)[0].tolist() + [i]  # all parents (1s) + self
            self.models.append(
                ADDSTCN(
                    target=i,
                    input_size=len(input_idx),
                    cuda=cuda,
                    **model_params
                )
            )
            setattr(self, f'input_idx_{i}', input_idx)  # save indices for forward pass

    def forward(self, x):
        """
        x: (B, T, N)
        returns: (B, T, N)
        """
        B, T, N = x.shape
        outputs = []
        for i, model in enumerate(self.models):
            input_idx = getattr(self, f'input_idx_{i}')
            x_input = x[:, :, input_idx]          # (B, T, len(input_idx))

            # Transpose to (B, C, T) as expected by ADDSTCN
            x_input = x_input.transpose(1, 2)     # (B, len(input_idx), T)

            out = model(x_input)                  # (B, T, 1) — ADDSTCN returns transposed back
            outputs.append(out)                   # append (B, T, 1)

        return th.cat(outputs, dim=-1)            # (B, T, N)