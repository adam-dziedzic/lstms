import argparse
import re
import torch

from rnn_model import RNNClassifier
from lstm_model import LSTMClassifier


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize(text):
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    return tokens if tokens else [UNK_TOKEN]


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_type = checkpoint["model_type"]
    vocab = checkpoint["vocab"]

    if model_type == "rnn":
        model = RNNClassifier(
            vocab_size=len(vocab),
            embed_dim=checkpoint["embed_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_classes=checkpoint["num_classes"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
            pad_idx=checkpoint["pad_idx"],
        )
    else:
        model = LSTMClassifier(
            vocab_size=len(vocab),
            embed_dim=checkpoint["embed_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_classes=checkpoint["num_classes"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint["dropout"],
            pad_idx=checkpoint["pad_idx"],
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, vocab


@torch.no_grad()
def predict(text, model, vocab, device):
    unk_idx = vocab[UNK_TOKEN]
    tokens = tokenize(text)
    ids = [vocab.get(tok, unk_idx) for tok in tokens]

    input_ids = torch.tensor([ids], dtype=torch.long).to(device)
    lengths = torch.tensor([len(ids)], dtype=torch.long).to(device)

    logits = model(input_ids, lengths)
    probs = torch.softmax(logits, dim=1)[0]
    pred = torch.argmax(probs).item()

    label_map = {0: "negative", 1: "positive"}
    return label_map[pred], probs[pred].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab = load_model(args.checkpoint, device)

    if args.text is not None:
        label, confidence = predict(args.text, model, vocab, device)
        print(f"Text: {args.text}")
        print(f"Prediction: {label} (confidence={confidence:.4f})")
    else:
        print("Type a sentence and press Enter. Empty input to quit.")
        while True:
            text = input(">> ").strip()
            if not text:
                break
            label, confidence = predict(text, model, vocab, device)
            print(f"Prediction: {label} (confidence={confidence:.4f})")


if __name__ == "__main__":
    main()