# environment.py
import numpy as np
import random

class HouseEnvironment:
    def __init__(self, house_id):
        self.house_id = house_id
        self.reset()

    def reset(self):
        """Her yeni simülasyon döngüsünde evi sıfırlar."""
        self.day = 1
        self.max_days = 200
        self.p_caught = 0.01  # Başlangıçtaki gerçek yakalanma olasılığı (%1)
        self.score = 0
        self.is_dead = False
        self.death_reason = None  # "cheese" (peynir) veya "caught" (ekmek yerken yakalanma)
        return self.get_observation()

    def get_observation(self):
        """
        Farenin gördüğü durum (State) bilgisini oluşturur.
        Gerçek olasılığı gizleyip rastgele genişlikte bir aralık verir.
        Geri dönüş vektörü: [Normalize_Gün, Risk_Alt_Sınır, Risk_Üst_Sınır]
        """
        # Sinir ağının kararlı çalışması için gün sayısını 0.0 ile 1.0 arasına normalize ediyoruz.
        day_normalized = self.day / float(self.max_days)
        
        # Sizin belirttiğiniz gibi olasılıklar arası uzaklık (%1 ile %10 arasında) rastgele belirlenir.
        width = random.uniform(0.01, 0.10)
        
        # Gerçek yakalanma olasılığının (p_caught), bu aralığın içinde kalmasını garanti ediyoruz.
        offset = random.uniform(0, width)
        
        L_t = max(0.0, self.p_caught - offset)
        U_t = min(1.0, self.p_caught + (width - offset))
        
        return np.array([day_normalized, L_t, U_t], dtype=np.float32)

    def step(self, action):
        """
        Fare bir karar verir ve çevre bu kararı işler.
        Action:
        0 -> Ekmek kırıntısı (Hayatta kalma mücadelesi)
        1 -> Peynir (Garanti 100 puan ve ölüm)
        """
        if self.is_dead or self.day > self.max_days:
            return self.get_observation(), 0, True, {"reason": self.death_reason, "score": self.score}

        reward = 0
        done = False

        if action == 1:
            # Peynir seçimi: Doğrudan +100 puan, kesin ölüm
            reward = 100.0
            self.score += 100
            self.is_dead = True
            self.death_reason = "cheese"
            done = True
        else:
            # Ekmek kırıntısı seçimi: Günlük hayatta kalma zarı atılır
            roll = random.random()
            if roll < self.p_caught:
                # Yakalandı ve öldü (Ekmek yerken yakalanırsa peynir puanı alamaz)
                reward = -10.0  # Ölmeyi cezalandırmak için negatif ödül
                self.is_dead = True
                self.death_reason = "caught"
                done = True
            else:
                # Survived: Günlük 1 puan kazanır
                reward = 1.0
                self.score += 1
                self.day += 1
                if self.day > self.max_days:
                    done = True
                    self.death_reason = "survived_max"
                else:
                    # Gerçek yakalanma olasılığını güncelliyoruz (Rastgele yürüyüş - drift)
                    # Ortalama 0.001 artış yönünde, hafif dalgalanmalı bir artış modeli
                    drift = np.random.normal(0.001, 0.0005)
                    self.p_caught = max(0.005, min(1.0, self.p_caught + drift))

        next_state = self.get_observation()
        return next_state, reward, done, {"reason": self.death_reason, "score": self.score}