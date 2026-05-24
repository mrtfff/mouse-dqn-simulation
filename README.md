# 🐭 Çok Ajanlı DQN Fare Simülasyonu (Multi-Agent Mouse DQN Simulation)

Bu proje, **Derin Q-Öğrenme (Deep Q-Learning - DQN)** algoritmasını kullanan 25 bağımsız farenin (ajanın), riskli ev ortamlarında hayatta kalma ve ödül maksimizasyonu mücadelelerini konu alan interaktif bir simülasyon ve görselleştirme aracıdır. **PySide6** ile geliştirilmiş zengin arayüzü sayesinde, yapay zekaların anlık kararlarını, gerçek zamanlı yapay sinir ağı (DQN) sinaps ağırlıklarını ve akranlar arası bilgi paylaşımını canlı olarak takip edebilirsiniz.

---

## 🚀 Öne Çıkan Özellikler

- **Çok Ajanlı (Multi-Agent) Mimari:** 25 bağımsız evde, 25 farklı fare aynı anda eğitilir ve kararlar alır.
- **Canlı Yapay Sinir Ağı (DQN) Görselleştirici:** İstediğiniz evin üzerine tıklayarak farenin yapay beynindeki (giriş, gizli katmanlar ve çıkış katmanı) sinaps ağırlıklarını, pozitif/negatif etkilerini ve anlık Q değerlerini canlı olarak gözlemleyebilirsiniz.
- **Dinamik Kişilik Filtreleri (Brave, Cowardly, Balanced):** Farelerin kararlarını etkileyen kişilik yapılarını (Cesur, Korkak, Dengeli) çalışma zamanında (runtime) sağ tıklayarak değiştirebilirsiniz.
- **P2P Bilgi Paylaşımı (Knowledge Sharing):** Her 10 episodda bir, en başarılı lider farenin sinir ağı ağırlıkları diğer farelere yumuşak bir şekilde aktarılır (soft sharing - $\tau=0.10$) ve hafıza havuzundaki (experience replay) deneyimler akranlarıyla paylaşılarak kolektif öğrenme hızlandırılır.
- **Gelişmiş Kontrol Paneli:** 
  - Simülasyon hızını milisaniye cinsinden ayarlama.
  - Tek adımda (gün bazlı) ilerleme veya otomatik mod.
  - Beyinleri diske kaydetme ve kaldığı yerden yükleme.
  - Ortalama skor gelişimini gösteren dinamik Matplotlib grafiği.

---

## 🧠 Karar Mekanizması ve Simülasyon Kuralları

Her fare, her gün iki eylemden birini seçmek zorundadır:
1. **Ekmek Kırıntısı (Action 0):** Hayatta kalma mücadelesidir. Fare günü sağ salim atlatırsa **+1 puan** kazanır. Ancak her gün yakalanma riski ($p_{caught}$) hafifçe artar (rastgele yürüyüş / drift modeli).
2. **Peynir (Action 1):** Fareye doğrudan **+100 puan** kazandırır ancak farenin kesin olarak ölümüyle (kapan) sonuçlanır.

### Durum Vektörü (State Space)
Farenin gözlemleyebildiği durum bilgisi 3 boyuttan oluşur:
$$\text{State} = [\text{Normalize Gün}, \text{Risk Alt Sınırı}, \text{Risk Üst Sınırı}]$$
*Fareye gerçek yakalanma olasılığı gizlenerek, her gün rastgele genişlikte (%1 ile %10 arasında) bir risk aralığı ($[L_t, U_t]$) sunulur.*

### Kişilik Modelleri
- **CESUR (Brave):** Keşif (exploration) odaklıdır. İlk 15 gün boyunca yakalanma riski ne olursa olsun ekmek kırıntısı yemeye zorlanır.
- **KORKAK (Coward):** Riskten kaçınma odaklıdır. Tahmini üst risk sınırı %5'in üzerine çıktığı an panikleyerek doğrudan peynire (Action 1) yönelir.
- **DENGELİ (Balanced):** Tamamen DQN sinir ağının ürettiği Q değerlerine (keşif ve sömürü dengesine) göre karar verir.

---

## 🛠️ Kurulum ve Çalıştırma

Proje Windows ve Python 3.10+ ortamlarında test edilmiştir.

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/mrtfff/mouse-dqn-simulation.git
cd mouse-dqn-simulation
```

### 2. Sanal Ortam Oluşturun ve Aktive Edin
**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 4. Simülasyonu Başlatın
```bash
python main.py
```

---

## 📁 Proje Dosya Yapısı

```text
mouse-dqn-simulation/
├── src/
│   ├── __init__.py
│   ├── agent.py         # PyTorch DQN Ajan mimarisi ve Deneyim Tekrarı (Replay Memory)
│   ├── environment.py   # Fare ortamı, ödül mekanizması ve risk hesaplama
│   ├── simulation.py    # Çoklu ajan yönetimi, P2P bilgi paylaşımı ve disk kayıt işlemleri
│   └── ui.py            # PySide6 Arayüz bileşenleri ve Ağ Görselleştirici (Network Painter)
├── saved_models/        # Eğitilen ajan ağırlıklarının ve simülasyon metasının kaydedildiği klasör
├── .gitignore           # Git tarafından takip edilmeyecek dosyalar listesi
├── LICENSE              # MIT Lisans belgesi
├── README.md            # Proje tanıtım ve kullanım kılavuzu
├── requirements.txt     # Gerekli Python kütüphaneleri listesi
└── main.py              # Projeyi başlatan ana giriş noktası
```

---

## ⚖️ Lisans

Bu proje **MIT Lisansı** altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına göz atabilirsiniz.
