# Fulox Basit RNN Dil Modeli
import torch
import torch.nn as nn

class SimpleRNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, output_dim=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size if output_dim is None else output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        logits = self.fc(output)
        return logits

# Test amaçlı örnek
if __name__ == "__main__":
    vocab_size = 100
    model = SimpleRNNModel(vocab_size)
    x = torch.randint(0, vocab_size, (1, 10))  # 1 örnek, 10 kelime
    out = model(x)
    print(out.shape)
