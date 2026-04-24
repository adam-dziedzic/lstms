import argparse
import random
import re
from collections import Counter

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from rnn_model import RNNClassifier
from lstm_model import LSTMClassifier


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text):
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    return tokens if tokens else [UNK_TOKEN]


def build_vocab(train_split, min_freq=2):
    counter = Counter()
    for example in train_split:
        counter.update(tokenize(example["sentence"]))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)

    return vocab


class SSTDataset(Dataset):
    def __init__(self, hf_split, vocab):
        self.vocab = vocab
        self.unk_idx = vocab[UNK_TOKEN]
        self.samples = []

        for example in hf_split:
            label = example["label"]
            if label == -1:
                continue

            ids = self.encode(example["sentence"])
            self.samples.append((torch.tensor(ids, dtype=torch.long), label))

    def encode(self, text):
        tokens = tokenize(text)
        return [self.vocab.get(tok, self.unk_idx) for tok in tokens]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def make_collate_fn(pad_idx):
    def collate_fn(batch):
        sequences, labels = zip(*batch)
        lengths = torch.tensor([len(x) for x in sequences], dtype=torch.long)
        padded = pad_sequence(sequences, batch_first=True, padding_value=pad_idx)
        labels = torch.tensor(labels, dtype=torch.long)
        return padded, lengths, labels

    return collate_fn


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for input_ids, lengths, labels in loader:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for input_ids, lengths, labels in loader:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

    return total_loss / total_count, total_correct / total_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rnn", "lstm"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--save_path", type=str, default="sst_model.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")
    train_split = dataset["train"]
    valid_split = dataset["validation"]

    print("Building vocabulary...")
    vocab = build_vocab(train_split, min_freq=args.min_freq)
    pad_idx = vocab[PAD_TOKEN]

    train_data = SSTDataset(train_split, vocab)
    valid_data = SSTDataset(valid_split, vocab)

    collate_fn = make_collate_fn(pad_idx)

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    if args.model == "rnn":
        model = RNNClassifier(
            vocab_size=len(vocab),
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_classes=2,
            num_layers=args.num_layers,
            dropout=args.dropout,
            pad_idx=pad_idx,
        )
    else:
        model = LSTMClassifier(
            vocab_size=len(vocab),
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_classes=2,
            num_layers=args.num_layers,
            dropout=args.dropout,
            pad_idx=pad_idx,
        )

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "model_type": args.model,
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "pad_idx": pad_idx,
                "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "num_classes": 2,
            }
            torch.save(checkpoint, args.save_path)
            print(f"Saved best model to {args.save_path}")

    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()