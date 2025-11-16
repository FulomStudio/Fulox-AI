# Fulox Inference (Cevap Üretme) Fonksiyonu
import torch
import sys
sys.path.append('./model')
from simple_rnn import SimpleRNNModel
from tokenizer import tokenize

# Vocab oluşturma fonksiyonu (train.py ile aynı)
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

def inverse_vocab(vocab):
    return {v: k for k, v in vocab.items()}

if __name__ == "__main__":
    filepath = "data/conversations.txt"
    vocab = build_vocab(filepath)
    inv_vocab = inverse_vocab(vocab)
    model = SimpleRNNModel(vocab_size=len(vocab))
    model.load_state_dict(torch.load("model/fulox_rnn.pth"))
    model.eval()

    # Kullanıcıdan giriş al
    input_text = input("Soru veya kelime girin: ")
    tokens = tokenize(input_text)
    if not tokens:
        print("Geçerli bir kelime girin.")
    else:
        input_token = vocab.get(tokens[-1], None)
        if input_token is None:
            print("Kelime vocabda yok.")
        else:
            x = torch.tensor([[input_token]])
            with torch.no_grad():
                output = model(x)
                next_token_id = torch.argmax(output[0,0]).item()
                next_word = inv_vocab.get(next_token_id, "(bilinmiyor)")
                print(f"Tahmini sonraki kelime: {next_word}")
