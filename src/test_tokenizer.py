import tiktoken

def main():
    print("=== Tokenizer Test Ekranı ===")
    print("ChatGPT'nin beyni (cl100k_base) kelimeleri nasıl anlıyor?")
    print("Çıkmak için 'q' veya 'quit' yazabilirsin.\n")
    
    enc = tiktoken.get_encoding("cl100k_base")
    
    while True:
        text = input("\nBir kelime veya cümle yaz: ")
        if text.lower() in ['q', 'quit', 'çıkış']:
            break
            
        if not text.strip():
            continue
            
        tokens = enc.encode(text)
        print(f"\nModelin Gördüğü Sayılar ({len(tokens)} token):")
        print(tokens)
        
        print("\nSayılara Denk Gelen Parçalar (Subwords):")
        for t in tokens:
            # Token'ın orijinal metindeki karşılığını göster
            piece = enc.decode([t])
            print(f"[{t}] -> '{piece}'")

if __name__ == "__main__":
    main()
