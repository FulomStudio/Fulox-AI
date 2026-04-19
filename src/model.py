import torch
import torch.nn as nn
from torch.nn import functional as F

# Hiperparametreler
vocab_size = 100277 
n_embd = 384        
n_head = 6          
n_layer = 6         
dropout = 0.1       
block_size = 128    

# 1. RMSNorm (Daha Hızlı ve Verimli Normalizasyon)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # x.pow(2).mean() varyansı bulur. RMSNorm sadece varyansla çalışarak hızı artırır.
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight

# 2. SwiGLU İleri Besleme Ağı (Llama'nın Düşünme Mekanizması)
class SwiGLU(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, in_dim, bias=False)
        self.w3 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Swish-Gated Linear Unit (SiLU aktivasyonu ile daha organik bağlar kurar)
        hidden = F.silu(self.w1(x)) * self.w3(x)
        return self.dropout(self.w2(hidden))

# 3. RoPE (Rotary Position Embedding) - Kelime Sırasını Kodlama Matematikleri
def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Kompleks sayılar
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # (1, T, 1, head_dim/2)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# 4. KV Cache Destekli Attention
class Attention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, freqs_cis, kv_cache=None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim)
        
        # RoPE (Döner Pozisyon) Ekle
        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        # KV Cache: Eğer eski kelimelerin hafızası varsa birleştir
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=1)
            v = torch.cat([v_cache, v], dim=1)
        new_kv_cache = (k, v)
            
        # Attention Çarpımları
        q = q.transpose(1, 2) # (B, n_head, T, head_dim)
        k = k.transpose(1, 2) # (B, n_head, T_cache, head_dim)
        v = v.transpose(1, 2) # (B, n_head, T_cache, head_dim)
        
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Maskeleme (Sadece ilk girdi cümlesi okurken (T>1) ileriye bakmayı engelleriz)
        if T > 1:
            mask = self.tril[:T, :T]
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        
        out = (probs @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(out), new_kv_cache

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attention = Attention(n_embd, n_head)
        # Llama standardında SwiGLU genisligi: n_embd * (8/3)
        self.feed_forward = SwiGLU(n_embd, int((8/3) * n_embd)) 
        self.rms1 = RMSNorm(n_embd)
        self.rms2 = RMSNorm(n_embd)

    def forward(self, x, freqs_cis, kv_cache=None):
        att_out, new_kv_cache = self.attention(self.rms1(x), freqs_cis, kv_cache)
        x = x + att_out
        x = x + self.feed_forward(self.rms2(x))
        return x, new_kv_cache

class FuloxAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.layers = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.norm = RMSNorm(n_embd)
        self.output = nn.Linear(n_embd, vocab_size, bias=False)
        
        # RoPE frekanslarını en baştan hesapla ve hafızada tut
        self.freqs_cis = precompute_freqs_cis(n_embd // n_head, block_size)

    def forward(self, tokens, targets=None, kv_caches=None):
        B, T = tokens.shape
        x = self.tok_emb(tokens)
        
        self.freqs_cis = self.freqs_cis.to(x.device)
        
        # KV Cache Kontrolü
        if kv_caches is None:
            # Eğitim veya ilk metin okunurken
            kv_caches = [None] * len(self.layers)
            freqs_cis = self.freqs_cis[:T]
        else:
            # Sadece tek kelime üretilirken (KV Cache devrede)
            # Şu anki kelimenin pozisyonunu bul
            start_pos = kv_caches[0][0].shape[1] 
            freqs_cis = self.freqs_cis[start_pos:start_pos+1]
            
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            x, new_cache = layer(x, freqs_cis, kv_caches[i])
            new_kv_caches.append(new_cache)
            
        x = self.norm(x)
        logits = self.output(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss, new_kv_caches

if __name__ == "__main__":
    print("--- Fulox AI V1.5 (Genişletilmiş Llama Mimari) Testi ---")
    m = FuloxAI()
    total_params = sum(p.numel() for p in m.parameters())
    print(f"Modelin Toplam Parametre Sayısı: {total_params / 1e6:.2f} Milyon")
    
    dummy_input = torch.randint(0, vocab_size, (1, 10))
    logits, loss, _ = m(dummy_input)
    print(f"Çıktı Şekli: {logits.shape}")
    print("\nV1.5 Mimari Testi Başarılı!")
