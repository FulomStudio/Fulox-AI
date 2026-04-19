import os
import sys
import requests
from bs4 import BeautifulSoup

def scrape_url(url):
    try:
        # Tarayıcı gibi davranmak için header ekliyoruz
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Sayfadaki paragrafları al
        paragraphs = soup.find_all('p')
        # Sadece çok kısa olmayan, anlamlı cümleleri birleştir
        text = "\n".join([p.get_text() for p in paragraphs if len(p.get_text()) > 30])
        return text
    except Exception as e:
        print(f"Hata: URL okunamadı - {e}")
        return ""

def main():
    if len(sys.argv) < 2:
        print("Kullanım: python src/scrape_and_learn.py <URL>")
        sys.exit(1)
        
    url = sys.argv[1]
    print(f"Veri çekiliyor: {url}")
    
    text = scrape_url(url)
    if text:
        dataset_path = "data/dataset.txt"
        os.makedirs("data", exist_ok=True)
        
        # Dosyanın sonuna veriyi ekle (append modunda)
        with open(dataset_path, "a", encoding="utf-8") as f:
            f.write("\n\n--- YENİ VERİ EKLENDİ ---\n\n")
            f.write(text)
            
        print(f"Başarılı! Çekilen metin (boyut: {len(text)} karakter) modele öğretilmek üzere veri setine eklendi.")
    else:
        print("Uyarı: Veri çekilemedi veya sayfa metin içermiyor.")

if __name__ == "__main__":
    main()
