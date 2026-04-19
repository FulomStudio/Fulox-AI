import os
import torch
import tiktoken
from model import FuloxAI, block_size

def generate_response(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8, device='cpu'):
    """ KV Cache destekli, Llama mimarisine uygun çok daha hızlı sohbet fonksiyonu """
    tokens = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0) 
    
    model.eval()
    kv_caches = None
    
    with torch.no_grad():
        # Prompt limitini kontrol et
        if idx.shape[1] > block_size - 1:
            idx = idx[:, -(block_size-1):]
            
        # İlk geçiş: Tüm promptu okuyup hafızayı (Cache) doldururuz
        logits, _, kv_caches = model(idx, kv_caches=None)
        
        # İlk kelime tahmini
        logits = logits[:, -1, :] / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Geri kalan kelimeleri üretme döngüsü
        for _ in range(max_new_tokens - 1):
            if idx.shape[1] >= block_size:
                # Eğer maksimum uzunluğu aşarsak cache'i sıfırlamak güvenlidir
                idx_cond = idx[:, -block_size:]
                logits, _, kv_caches = model(idx_cond, kv_caches=None)
            else:
                # İŞTE HIZ BURADAN GELİYOR!
                # KV Cache sayesinde tüm cümleyi değil, SADECE ürettiğimiz son 1 kelimeyi modele veriyoruz!
                logits, _, kv_caches = model(idx_next, kv_caches=kv_caches)
                
            logits = logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
    generated_tokens = idx[0].tolist()
    return tokenizer.decode(generated_tokens)

def main():
    print("--- Fulox AI V1.5 (Genişletilmiş Llama Mimari) Konsol Arayüzü ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint_path = "checkpoints/fulox_v1.5_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Hata: {checkpoint_path} bulunamadı! V1.5 Mimariye geçtik, lütfen modeli YENİDEN EĞİTİN (python src/train.py).")
        return
        
    print("Model ağırlıkları yükleniyor...")
    model = FuloxAI()
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except Exception as e:
        print(f"UYARI: Eski versiyonun ağırlıkları yeni (V1.5) mimari ile uyumsuz. Lütfen modeli baştan eğitin.")
        print("Komut: python src/train.py")
        return
        
    model.to(device)
    
    print("Tokenizer yükleniyor...")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    print("\nSistem Hazır! Fulox AI V1.5 ile iletişime geçebilirsiniz. (Çıkış için 'q' veya 'quit' yazın)")
    print("-" * 50)
    
    while True:
        user_input = input("\nKullanıcı: ")
        if user_input.lower() in ['q', 'quit', 'çıkış']:
            break
            
        if not user_input.strip():
            continue
            
        print("Fulox AI V1.5:", end=" ", flush=True)
        full_response = generate_response(model, tokenizer, user_input, max_new_tokens=50, temperature=0.7, device=device)
        
        only_new_text = full_response[len(user_input):]
        print(only_new_text)

if __name__ == "__main__":
    main()
