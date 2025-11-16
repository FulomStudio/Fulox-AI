# Fulox Eğitim Döngüsü ve Veri Yükleyici
import torch
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('./model')
from simple_rnn import SimpleRNNModel
from tokenizer import tokenize

class TextDataset(Dataset):
    def __init__(self, filepath, vocab):
        self.samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                tokens = tokenize(line)
                if len(tokens) > 1:
                    # Her kelimeyi bir sonraki kelimeyi tahmin etmek için kullan
                    for i in range(len(tokens)-1):
                        input_token = vocab[tokens[i]]
                        target_token = vocab[tokens[i+1]]
                        self.samples.append((input_token, target_token))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx][0]), torch.tensor(self.samples[idx][1])

def build_vocab(filepath):
    vocab = {}
    idx = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            tokens = tokenize(line)
            for token in tokens:
                if token not in vocab:
                    vocab[token] = idx
                    idx += 1
    return vocab

if __name__ == "__main__":
    filepath = "data/conversations.txt"
    vocab = build_vocab(filepath)
    dataset = TextDataset(filepath, vocab)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SimpleRNNModel(vocab_size=len(vocab))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Basit eğitim döngüsü
    for epoch in range(5):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs = inputs.unsqueeze(1)  # (batch, seq_len=1)
            outputs = model(inputs)
            outputs = outputs.squeeze(1)  # (batch, vocab_size)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), "model/fulox_rnn.pth")
    print("Model kaydedildi: model/fulox_rnn.pth")
