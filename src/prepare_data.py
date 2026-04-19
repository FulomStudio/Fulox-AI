import os
from datasets import load_dataset
from tqdm import tqdm

def main():
    print("--- Fulox AI V1 Büyük Veri (Big Data) Hazırlayıcı ---")
    print("HuggingFace sunucularına bağlanılıyor...")
    
    # RAM'i patlatmamak için streaming=True kullanıyoruz.
    # Böylece Wikipedia'nın tamamı inmeyecek, makaleler okundukça diske yazılıp RAM'den silinecek.
    try:
        dataset = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train", streaming=True)
    except Exception as e:
        print(f"Hata oluştu: Sunucuya bağlanılamadı. ({e})")
        return
    
    target_size_mb = 100
    target_size_bytes = target_size_mb * 1024 * 1024
    current_size = 0
    
    dataset_path = "data/dataset.txt"
    os.makedirs("data", exist_ok=True)
    
    print(f"Hedef: {target_size_mb} MB saf metin. İndirme ve yazma işlemi başlıyor...")
    
    # Progress bar (İlerleme Çubuğu)
    pbar = tqdm(total=target_size_mb, desc="İndirilen Veri (MB)", unit="MB")
    
    with open(dataset_path, "w", encoding="utf-8") as f:
        for data in dataset:
            text = data['text']
            
            # Eğer makale çok kısaysa (örn: sadece başlık) atla
            if len(text) < 500:
                continue
                
            f.write(text + "\n\n")
            
            # Dosyaya eklenen bayt sayısını hesapla
            added_bytes = len(text.encode('utf-8')) + 2
            current_size += added_bytes
            
            # Progress bar'ı MB cinsinden ilerlet
            pbar.update(added_bytes / (1024 * 1024))
            
            # 100 MB'a ulaşınca durdur
            if current_size >= target_size_bytes:
                break
                
    pbar.close()
    print(f"\nŞahane! Veri seti başarıyla {target_size_mb} MB boyutuna ulaştı.")
    print(f"Dosya hazır: {dataset_path}")
    print("\nSıradaki Adım: Lütfen 'python src/tokenizer.py' komutunu çalıştırarak bu metni sayılara çevir.")

if __name__ == "__main__":
    main()
