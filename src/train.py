import torch
import os
from model import FuloxAI, block_size, vocab_size

# Eğitim Ayarları
batch_size = 8         # Aynı anda eğitilecek cümle parçası sayısı
max_iters = 50000      # Toplam eğitim turu sayısı (Gece boyu eğitim için 50000'e çıkardık)
eval_interval = 100    # Çok yavaşladığı için her 100 adımda bir ekrana yazdıracağız
learning_rate = 1e-3   # Öğrenme hızı
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Verileri Yükle
print("Tokenize edilmiş veriler yükleniyor...")
train_data = torch.load("data/train.pt")
val_data = torch.load("data/val.pt")

def get_batch(split):
    # Veriden rastgele bloklar (cümleler) çek
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # Hedef: Bir kelime sonrası
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    # Modeli test edip başarısını ölç
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(50)
        for k in range(50):
            X, Y = get_batch(split)
            logits, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    print(f"\n--- Fulox AI Eğitim Sistemi Başlatıldı ---")
    print(f"Kullanılan Donanım: {device.upper()}")
    
    # Modeli Başlat
    print("Fulox AI modeli başlatılıyor...")
    model = FuloxAI()
    
    # Kaldığı Yerden Devam Etme (Resume) Mantığı
    checkpoint_path = "checkpoints/fulox_v1.5_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"Önceki eğitim bulundu! Ağırlıklar yükleniyor... ({checkpoint_path})")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("Sıfırdan yepyeni bir beyin oluşturuluyor...")
        
    model = model.to(device)
    
    # Optimizasyon
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print("\nEğitim süreci başlıyor...")
    print("-" * 50)
    
    best_val_loss = float('inf')
    os.makedirs("checkpoints", exist_ok=True)
    
    for iter in range(max_iters):
        # Arada bir durumu ekrana yazdır
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"Adım {iter:5d}/{max_iters} | Eğitim Hatası (Loss): {losses['train']:.4f} | Test Hatası: {losses['val']:.4f}")
            
            # Ezberlemeyi Önleme (Save Best Model)
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  -> Yeni en iyi model diske kaydedildi! (Test Hatası: {best_val_loss:.4f})")
                
        # Veri çek
        xb, yb = get_batch('train')
        
        # İleri ve Geri besleme (Gerçek öğrenme burası)
        logits, loss, _ = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    print("-" * 50)
    print(f"Eğitim Tamamlandı! Elde edilen en iyi Test Hatası: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
