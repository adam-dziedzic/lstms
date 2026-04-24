import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=128,
        num_classes=2,
        num_layers=1,
        dropout=0.2,
        pad_idx=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, lengths):
        # input_ids: [batch, seq_len]
        embedded = self.dropout(self.embedding(input_ids))  # [batch, seq_len, embed_dim]

        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (h_n, c_n) = self.lstm(packed)
        # h_n: [num_layers, batch, hidden_dim]
        last_hidden = h_n[-1]  # [batch, hidden_dim]

        logits = self.fc(self.dropout(last_hidden))
        return logits