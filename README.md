# LSTMs

The simple code to create, train, and run RNNs and LSTMs. The goal is to do sentiment analysis for movie reviews (many to one mapping).

### Structure

rnn_model.py — a basic RNN classifier

lstm_model.py — a basic LSTM classifier

train_sst.py — training on SST-2 sentiment classification

infer.py — inference on new sentences

### Install

`pip install torch datasets`

### Train the RNN

```bash
python train_sst.py --model rnn --epochs 5 --save_path sst_rnn.pt
```

### Train the LSTM

```bash
python train_sst.py --model lstm --epochs 5 --save_path sst_lstm.pt
```

---

# How to run inference

### With the RNN model

```bash
python infer.py --checkpoint sst_rnn.pt --text "this movie was wonderful"
```

### With the LSTM model

```bash
python infer.py --checkpoint sst_lstm.pt --text "this movie was wonderful"
```

You can also start interactive mode:

```bash
python infer.py --checkpoint sst_lstm.pt
```

Then type sentences like:

```text
>> this film is amazing
>> the plot was boring and predictable
```

---

# Example training output

I cannot run training here, but your console will look something like this:

```text
Using device: cuda
Loading SST-2 dataset...
Building vocabulary...
Epoch 01/5 | train_loss=0.6012 train_acc=0.6714 | val_loss=0.4798 val_acc=0.7805
Saved best model to sst_lstm.pt
Epoch 02/5 | train_loss=0.4301 train_acc=0.8052 | val_loss=0.3971 val_acc=0.8211
Saved best model to sst_lstm.pt
Epoch 03/5 | train_loss=0.3644 train_acc=0.8420 | val_loss=0.3812 val_acc=0.8314
Saved best model to sst_lstm.pt
...
Best validation accuracy: 0.83xx
```

---

# Example inference output

```bash
python infer.py --checkpoint sst_lstm.pt --text "this movie was surprisingly touching and funny"
```

Example output:

```text
Text: this movie was surprisingly touching and funny
Prediction: positive (confidence=0.9521)
```

Another:

```bash
python infer.py --checkpoint sst_lstm.pt --text "the film was dull and too long"
```

Example output:

```text
Text: the film was dull and too long
Prediction: negative (confidence=0.9184)
```

---

# Notes

- This example uses SST-2, which is binary sentiment classification:
  - `0 = negative`
  - `1 = positive`
- The LSTM usually works better than a plain RNN on text.
- This is intentionally simple:
  - basic tokenizer
  - learned embeddings
  - one recurrent model
  - linear classifier on top
