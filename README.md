# Fulox AI V1.5 🚀

Fulox AI, sıfırdan **PyTorch** kullanılarak geliştirilmiş, tamamen yerli ve açık kaynaklı bir Büyük Dil Modeli (LLM) projesidir. Klasik RNN mimarisinden çıkarılıp günümüzün en güçlü yapay zekalarında (Llama 3, Mistral, GPT-4) kullanılan modern **Transformer (Decoder-Only)** mimarisine geçiş yapılmıştır.

## 🌟 Öne Çıkan Özellikler (V1.5 Mimarisi)
* **15 Milyon Parametrelik Kapasite:** V1 prototipine kıyasla 15 kat daha büyük düşünme (mantık kurma) kapasitesi (`n_embd=384`, `n_layer=6`).
* **Erken Durdurma (Early Stopping):** Model eğitim sırasında ezberlemeye (Overfitting) başlarsa bunu tespit eder ve sadece "Test Hatasının en düşük olduğu en zeki anı" diske kaydeder.
* **Kesintisiz Eğitim (Resume):** Eğitim yarıda kesilse bile sıfırlanmaz, kaldığı yerden (`fulox_v1.5_model.pth`) ağırlıkları yükleyerek eski bilgisinin üstüne koymaya devam eder.
* **RoPE (Rotary Position Embeddings):** Modelin kelimelerin göreceli sırasını kusursuz anlaması için matematiksel döner pozisyon kodlaması.
* **SwiGLU Aktivasyonu:** Klasik ReLU yerine çok daha verimli öğrenme sağlayan, Llama standartlarındaki "Düşünme Katmanı".
* **RMSNorm & KV Cache:** Hızlı matematiksel normalizasyon ve sohbet sırasında kelime üretim (Inference) hızını artıran yapay zeka standartları.
* **BPE Tokenizer & Streaming:** OpenAI `tiktoken` altyapısı ve HuggingFace üzerinden 100 MB+ veriyi RAM'i doldurmadan işleme yeteneği.

## ⚙️ Kurulum ve Kullanım

### 1. Gereksinimler
Projeyi kendi bilgisayarınızda çalıştırmak için **Python 3.8+** ve aşağıdaki kütüphaneler gereklidir. Terminalinize şu komutu yazarak gereksinimleri kurun:
```bash
pip install -r requirements.txt
```

### 2. Veri Seti Hazırlama (100 MB Türkçe Wikipedia)
Modelin zekasını oluşturacak metinleri internetten (HuggingFace üzerinden) otomatik olarak indirmek için:
```bash
python src/prepare_data.py
```
*(Bu adım internet hızınıza bağlı olarak birkaç dakika sürebilir. İndirme bittiğinde 100 MB boyutunda saf Türkçe bir metin dosyanız olacaktır.)*

### 3. Tokenizasyon (Metinleri Sayılara Çevirme)
İndirilen metni yapay zekanın anlayacağı Tensör (`.pt`) dosyalarına çevirmek için:
```bash
python src/tokenizer.py
```

### 4. Modeli Eğitme (Training)
Tensör verilerini okutarak beyni (ağırlıkları) oluşturmak için:
```bash
python src/train.py
```
*(Eğitim süresi bilgisayarınızın işlemci/ekran kartı gücüne göre değişebilir. `max_iters` değeri 50.000 olarak ayarlanmıştır. İstediğiniz an `CTRL+C` ile eğitimi durdurabilirsiniz, Early Stopping özelliği sayesinde o ana kadar elde edilen en iyi model otomatik olarak diske kaydedilmiş olacaktır.)*

### 5. Sohbet Arayüzü (Chat)
Eğitim bittikten sonra ağırlıklar (checkpoints) kaydedilir. Kendi oluşturduğunuz bu zeka ile terminal üzerinden anında sohbet etmek için:
```bash
python src/chat.py
```

---
*Furkan Yeşilrımak:* [@fulox](https://github.com/fulox) | Açık kaynak kodlu ve eğitim amaçlı modern bir yapay zeka çekirdek projesidir. Her türlü katkıya açıktır.
