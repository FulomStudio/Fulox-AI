# Fulox Tokenizer
# Metni kelimelere ayıran basit bir tokenizer

def tokenize(text):
    # Noktalama işaretlerini ve özel karakterleri ayıkla
    import re
    text = re.sub(r"[\.,!?;:\-\"']", "", text)
    # Küçük harfe çevir
    text = text.lower()
    # Kelimelere ayır
    tokens = text.split()
    return tokens

if __name__ == "__main__":
    with open("data/conversations.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        print(tokenize(line))
