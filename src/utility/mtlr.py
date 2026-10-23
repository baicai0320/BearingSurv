import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
from typing import List, Tuple, Optional, Union
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import trange
from torch.utils.data import DataLoader, TensorDataset
from utility.survival import reformat_survival
from utility.loss import mtlr_nll

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

class mtlr(nn.Module):
    """Multi-task logistic regression for individualised
    survival prediction.

    The MTLR time-logits are computed as:
    `$z = sum_k x^T w_k + b_k$`,
    where `$w_k$` and `$b_k$` are learnable weights and biases for each time
    interval.

    Note that a slightly more efficient reformulation is used here, first
    proposed in [2]_.

    References
    ----------
    ..[1] C.-N. Yu et al., ‘Learning patient-specific cancer survival
    distributions as a sequence of dependent regressors’, in Advances in neural
    information processing systems 24, 2011, pp. 1845–1853.
    ..[2] P. Jin, ‘Using Survival Prediction Techniques to Learn
    Consumer-Specific Reservation Price Distributions’, Master's thesis,
    University of Alberta, Edmonton, AB, 2015.
    """

    def __init__(self, in_features: int, num_time_bins: int, config: argparse.Namespace):
        """Initialises the module.

        Parameters
        ----------
        in_features
            Number of input features.
        num_time_bins
            The number of bins to divide the time axis into.
        """
        super().__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
        self.config = config
        self.in_features = in_features
        self.num_time_bins = num_time_bins + 1  # + extra time bin [max_time, inf)

        self.mtlr_weight = nn.Parameter(torch.Tensor(self.in_features,
                                                     self.num_time_bins - 1))
        self.mtlr_bias = nn.Parameter(torch.Tensor(self.num_time_bins - 1))

        # `G` is the coding matrix from [2]_ used for fast summation.
        # When registered as buffer, it will be automatically
        # moved to the correct device and stored in saved
        # model state.
        self.register_buffer(
            "G",
            torch.tril(
                torch.ones(self.num_time_bins - 1,
                           self.num_time_bins,
                           requires_grad=True)))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass on a batch of examples.

        Parameters
        ----------
        x : torch.Tensor, shape (num_samples, num_features)
            The input data.

        Returns
        -------
        torch.Tensor, shape (num_samples, num_time_bins - 1)
            The predicted time logits.
        """
        out = torch.matmul(x, self.mtlr_weight) + self.mtlr_bias
        return torch.matmul(out, self.G)

    def reset_parameters(self):
        """Resets the model parameters."""
        nn.init.xavier_normal_(self.mtlr_weight)
        nn.init.constant_(self.mtlr_bias, 0.)

    def __repr__(self):
        return (f"{self.__class__.__name__}(in_features={self.in_features},"
                f" num_time_bins={self.num_time_bins})")

    def get_name(self):
        return self._get_name()

class DeepMTLR(nn.Module):
    """
    Deep Multi-task logistic regression (Deep-MTLR).
    Uses an MLP encoder to capture non-linear relationships before the MTLR head.
    """

    def __init__(self, in_features: int, num_time_bins: int, config: argparse.Namespace):
        super().__init__()
        if num_time_bins < 1:
            raise ValueError("The number of time bins must be at least 1")
        if in_features < 1:
            raise ValueError("The number of input features must be at least 1")
            
        self.config = config
        self.in_features = in_features
        self.num_time_bins = num_time_bins + 1 

        # 1. 获取超参数 (现有代码中已经有了这些 params)
        hidden_size = getattr(config, 'hidden_size', 32)
        dropout_rate = getattr(config, 'dropout', 0.25)

        # 2. 定义非线性特征提取器 (MLP)
        # 结构: Input -> Linear -> ReLU -> Dropout -> Hidden Features
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )

        # 3. MTLR Head (从 Hidden Size 映射到 Time Bins)
        # 注意：这里的输入维度变成了 hidden_size，而不是 in_features
        self.mtlr_weight = nn.Parameter(torch.Tensor(hidden_size,
                                                     self.num_time_bins - 1))
        self.mtlr_bias = nn.Parameter(torch.Tensor(self.num_time_bins - 1))

        # 4. Coding Matrix G (保持原样)
        self.register_buffer(
            "G",
            torch.tril(
                torch.ones(self.num_time_bins - 1,
                           self.num_time_bins,
                           requires_grad=True)))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A. 先通过 MLP 提取非线性特征
        embedding = self.feature_extractor(x)
        
        # B. 再进入 MTLR 线性层
        out = torch.matmul(embedding, self.mtlr_weight) + self.mtlr_bias
        
        # C. 最后乘上累积矩阵 G
        return torch.matmul(out, self.G)

    def reset_parameters(self):
        """初始化参数，包括新的 MLP 层"""
        # 初始化 MLP
        for layer in self.feature_extractor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        
        # 初始化 MTLR Head
        nn.init.xavier_normal_(self.mtlr_weight)
        nn.init.constant_(self.mtlr_bias, 0.)

    def get_name(self):
        return "DeepMTLR"

def mtlr_survival(
        logits: torch.Tensor,
        with_sample: bool = True
) -> torch.Tensor:
    """Generates predicted survival curves from predicted logits.

    Parameters
    ----------
    logits
        Tensor with the time-logits (as returned by the MTLR module)
        with size (n_samples, n_data, n_bins) or (n_data, n_bins).

    Returns
    -------
    torch.Tensor
        The predicted survival curves for each row in `pred` at timepoints used
        during training.
    """
    # TODO: do not reallocate G in every call
    if with_sample:
        assert logits.dim() == 3, "The logits should have dimension with with size (n_samples, n_data, n_bins)"
        G = torch.tril(torch.ones(logits.shape[2], logits.shape[2])).to(logits.device)
        density = torch.softmax(logits, dim=2)
        G_with_samples = G.expand(density.shape[0], -1, -1)

        # b: n_samples; i: n_data; j: n_bin; k: n_bin
        return torch.einsum('bij,bjk->bik', density, G_with_samples)
    else:   # no sampling
        assert logits.dim() == 2, "The logits should have dimension with with size (n_data, n_bins)"
        G = torch.tril(torch.ones(logits.shape[1], logits.shape[1])).to(logits.device)
        density = torch.softmax(logits, dim=1)
        return torch.matmul(density, G)

def make_mtlr_prediction(
        model: mtlr,
        x: torch.Tensor,
        time_bins: NumericArrayLike,
        config: argparse.Namespace
):
    model.eval()
    start_time = datetime.now()
    with torch.no_grad():
        pred = model.forward(x)
        end_time = datetime.now()
        inference_time = end_time - start_time
        survival_curves = mtlr_survival(pred, with_sample=False)
        # Tail probabilities from softmax sum to one analytically. Float32
        # accumulation can exceed one by a few ulps, which must be corrected
        # at model-output time rather than silently repaired by the evaluator.
        survival_curves = torch.clamp(survival_curves, min=0.0, max=1.0)

    time_bins = torch.cat([torch.tensor([0]), time_bins], dim=0).to(survival_curves.device)
    return survival_curves, time_bins, survival_curves.unsqueeze(0).repeat(config.n_samples_test, 1, 1)

def train_mtlr_model(
        model: nn.Module,
        data_train: pd.DataFrame,
        data_val: pd.DataFrame,
        time_bins: NumericArrayLike,
        config: dotdict,
        random_state: int,
        reset_model: bool = True,
        device: torch.device = torch.device("cuda")
) -> nn.Module:
    if config.verbose:
        print(f"Training {model.get_name()}: reset mode is {reset_model}, number of epochs is {config.num_epochs}, "
              f"learning rate is {config.lr}, C1 is {config.c1}, "
              f"batch size is {config.batch_size}, device is {device}.")
    train_size = data_train.shape[0]
    val_size = data_val.shape[0]
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if reset_model:
        model.reset_parameters()

    model = model.to(device)
    model.train()
    best_val_nll = np.inf
    best_ep = -1

    pbar = trange(config.num_epochs, disable=not config.verbose)

    start_time = datetime.now()
    x, y = reformat_survival(data_train, time_bins)
    x_val, y_val = reformat_survival(data_val, time_bins)
    x_val, y_val = x_val.to(device), y_val.to(device)
    train_loader = DataLoader(TensorDataset(x, y), batch_size=config.batch_size, shuffle=True)
    for i in pbar:
        nll_loss = 0
        for xi, yi in train_loader:
            xi, yi = xi.to(device), yi.to(device)
            optimizer.zero_grad()
            y_pred = model.forward(xi)
            loss = mtlr_nll(y_pred, yi, model, C1=config.c1, average=False)

            loss.backward()
            optimizer.step()

            nll_loss += (loss / train_size).item()
        logits_outputs = model.forward(x_val)
        eval_nll = mtlr_nll(logits_outputs, y_val, model, C1=0, average=True)
        pbar.set_description(f"[epoch {i + 1: 4}/{config.num_epochs}]")
        pbar.set_postfix_str(f"nll-loss = {nll_loss:.4f}; "
                                f"Validation nll = {eval_nll.item():.4f};")
        if config.early_stop:
            if best_val_nll > eval_nll:
                best_val_nll = eval_nll
                best_ep = i
            if (i - best_ep) > config.patience:
                break
    end_time = datetime.now()
    training_time = end_time - start_time
    # model.eval()
    return model


def train_mtlr_fixed_epochs(
        model: nn.Module,
        data_train: pd.DataFrame,
        time_bins: NumericArrayLike,
        config: dotdict,
        random_state: int,
        reset_model: bool = True,
        device: torch.device = torch.device("cuda")
) -> nn.Module:
    """Fit MTLR on every supplied row for exactly ``config.num_epochs``.

    The outer-CV runner uses this for its fixed-hyperparameter legacy protocol.
    It intentionally has no validation split, checkpoint selection, or early
    stopping: all outer-development bearings contribute gradients on every
    epoch of the final fit.
    """

    if data_train.empty:
        raise ValueError("MTLR fixed fit requires at least one training row")
    num_epochs = int(config.num_epochs)
    if num_epochs <= 0:
        raise ValueError("MTLR fixed fit requires num_epochs > 0")

    np.random.seed(int(random_state))
    torch.manual_seed(int(random_state))
    if reset_model:
        model.reset_parameters()

    x_train, y_train = reformat_survival(data_train, time_bins)
    batch_size = int(config.batch_size)
    if batch_size <= 0:
        raise ValueError("MTLR fixed fit requires batch_size > 0")
    generator = torch.Generator()
    generator.manual_seed(int(random_state))
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    model = model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=float(config.lr))
    final_loss = float("nan")
    for _ in range(num_epochs):
        cumulative_loss = 0.0
        for xi, yi in train_loader:
            xi, yi = xi.to(device), yi.to(device)
            optimizer.zero_grad()
            loss = mtlr_nll(
                model.forward(xi), yi, model, C1=float(config.c1), average=False
            )
            loss.backward()
            optimizer.step()
            cumulative_loss += float(loss.detach().item())
        final_loss = cumulative_loss / float(len(data_train))

    model.eval()
    model.fixed_fit_epochs_ = num_epochs
    model.fixed_fit_row_count_ = int(len(data_train))
    model.fixed_fit_final_loss_ = final_loss
    model.fixed_fit_early_stopping_ = False
    return model
