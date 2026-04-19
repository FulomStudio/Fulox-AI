import os
import torch
import tiktoken

def main():
    dataset_path = "data/dataset.txt"
    if not os.path.exists(dataset_path):
        print(f"Hata: {dataset_path} bulunamadı! Önce prepare_data.py çalıştırmalısın.")
        return

    print("Veri seti okunuyor...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Toplam karakter sayısı: {len(text):,}")

    # ChatGPT'nin kullandığı modern BPE tokenizer
    print("Tokenizer yükleniyor (tiktoken - cl100k_base)...")
    enc = tiktoken.get_encoding("cl100k_base")

    # Tüm metni sayılara (token) çevir
    print("Metin sayılara (token) dönüştürülüyor...")
    tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
    
    print(f"Toplam token sayısı: {len(tokens):,}")
    
    # Eğitim döngüsünde kullanabilmek için PyTorch Tensor formatına çevir
    data = torch.tensor(tokens, dtype=torch.long)
    
    # Modelin ne kadar iyi öğrendiğini test etmek için veriyi 
    # %90 Eğitim (Train) ve %10 Doğrulama (Validation) olarak ikiye böl
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # Tensor dosyalarını diske kaydet
    train_path = "data/train.pt"
    val_path = "data/val.pt"
    
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    
    print("-" * 30)
    print(f"Eğitim verisi kaydedildi: {train_path} ({len(train_data):,} token)")
    print(f"Doğrulama verisi kaydedildi: {val_path} ({len(val_data):,} token)")
    print("Tokenizasyon işlemi başarıyla tamamlandı! Model eğitimine hazırız.")

if __name__ == "__main__":
    main()
