"""
net.py — Neural network definition.

NOTE: PyTorch was unavailable at prototype time (no disk space for install on
this machine). sklearn.neural_network.MLPClassifier implements the same
feedforward architecture with backpropagation — the learning mechanics are
identical. Swap for a torch.nn.Module when PyTorch is available; the training
loop in supervised.py documents the equivalent PyTorch code inline.

Architecture:
    Input (TF-IDF features, up to 5000)
        → Dense(256) → ReLU → Dropout(0.3)    [layer 1]
        → Dense(128) → ReLU → Dropout(0.3)    [layer 2]
        → Dense(6)   → Softmax                [output: one unit per class]

Hyperparameters:
    optimizer   : Adam
    learning rate: 1e-3 (with adaptive schedule via sklearn's 'adaptive')
    regularisation: L2 weight decay α=1e-4
    max epochs  : 300 (early stopping via validation fraction)
    batch size  : 32
"""

from sklearn.neural_network import MLPClassifier


# ── Equivalent PyTorch definition (for reference / future use) ─────────────────
#
# import torch.nn as nn
#
# class ClassifierNet(nn.Module):
#     def __init__(self, input_dim: int, n_classes: int = 6):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, n_classes),
#         )
#     def forward(self, x):
#         return self.net(x)   # CrossEntropyLoss expects raw logits


def build_network() -> MLPClassifier:
    """
    Returns an untrained MLPClassifier with the architecture above.

    solver='adam'         — Adam optimiser
    learning_rate_init    — initial lr (same as torch Adam default)
    learning_rate='adaptive' — halves lr when training loss stops improving
    alpha                 — L2 regularisation (weight decay)
    batch_size            — mini-batch size
    max_iter              — maximum epochs
    early_stopping=True   — holds out 10% of training data as internal val set;
                            stops when val score doesn't improve for n_iter_no_change
    n_iter_no_change=20   — patience
    """
    return MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=32,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=False,
    )


if __name__ == "__main__":
    net = build_network()
    print("Network definition:")
    print(net)
