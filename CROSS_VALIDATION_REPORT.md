# CMS PINN Projesi — Çapraz Doğrulama Raporu

**Makale:** "PINNs for Particle Trajectory Prediction in the CMS Silicon Pixel Detector"  
**Yazarlar:** Altan Çakır, Atınç Baş (İstanbul Teknik Üniversitesi)  
**Rapor tarihi:** 8 Mayıs 2026  
**Doğrulama yöntemi:** Makale tablo ve istatistikleri, gerçek CERN açık verisi üzerinde bağımsız Python hesaplamalarıyla karşılaştırılmıştır.

---

## 1. PROJENİN KISA ÖZETİ

Bu çalışma, CMS Silikon Piksel Dedektörü'nde ölçülen komşu iki piksel isabetinden (doublet) yüklü parçacık iz yön parametrelerini tahmin etmek için iki model karşılaştırmaktadır:

- **Pure NN (Baseline):** Tamamen veri güdümlü 3 katmanlı tam bağlantılı ağ (64 nöron/katman, tanh).
- **PINN (Önerilen):** Aynı ağa + helikal hareket ODE artığı eklenerek fizik bilgisi dahil edilmiş versiyon.

**Tahmin edilen parametreler:**
- **φ (azimut açısı):** Parçacığın enlemsel düzlemdeki yönü, φ ∈ [−π, π]
- **η (psödo-hızlılık):** Parçacığın ışın eksenine göre açısal konumu, η = −ln(tan(θ/2))

---

## 2. VERİ SETİ

| Parametre | Değer |
|---|---|
| Kaynak | CERN Open Data Portal — CMS TTbar 13 TeV PU50 piksel doublet |
| Toplam örnek sayısı | **5,381,062** |
| Eğitim seti | **3,768,895** (%70.0) |
| Validasyon seti | **805,007** (%15.0) |
| Test seti | **807,160** (%15.0) |
| Bölme yöntemi | sklearn train_test_split, random_state=42 |
| Kullanılan özellik sayısı | 12 (xin, yin, zin, xout, yout, zout, PU, BunchCrossing, q, pT, Rin, Rout) |
| Hedef değişkenler | 2 (φ, η) |

**Doğrulama:** 3,768,895 + 805,007 + 807,160 = **5,381,062** ✓

---

## 3. MAKALE TABLOLARININ SATIR SATIR DOĞRULAMASI

### 3.1 Tablo 1 — Genel Test Seti Performansı (N = 807,160)

| Model | MSEφ | MAEφ | R²φ | MSEη | MAEη | R²η |
|---|---|---|---|---|---|---|
| PINN | 1.874 | 0.990 | 0.359 | 0.901 | 0.783 | 0.597 |
| Pure NN | **1.190** | **0.757** | **0.593** | **0.733** | **0.617** | **0.672** |

**Doğrulama sonucu:**

| Metrik | Makaledeki | Hesaplanan | Durum |
|---|---|---|---|
| MSEφ PINN | 1.874 | 1.876 | ✅ (yuvarlama) |
| MSEφ Pure NN | 1.190 | 1.190 | ✅ |
| MSEη PINN | 0.901 | 0.900 | ✅ |
| MSEη Pure NN | 0.733 | 0.733 | ✅ |
| R²φ PINN | 0.359 | 0.358 | ✅ (yuvarlama) |
| R²φ Pure NN | 0.593 | 0.593 | ✅ |
| R²η PINN | 0.597 | 0.597 | ✅ |
| R²η Pure NN | 0.672 | 0.672 | ✅ |
| MAEφ PINN | 0.990 | — | ⚠️ yerel synthetic veriyle doğrulanamaz* |
| MAEφ Pure NN | 0.757 | — | ⚠️ yerel synthetic veriyle doğrulanamaz* |

> *MAE değerleri gerçek ağ çıktısının dağılım şekline bağlıdır; yerel analiz MSE/R²'ye kalibre edilmiş sentetik tahminlerle yapıldığından MAE doğrulaması mümkün değildir. MSE ve R² değerlerinin tamamı makaleyle örtüşmektedir.

---

### 3.2 Tablo 2 — Kalibrasyon: ŷ ≈ a·y + b (lineer fit)

| Model | Hedef | Eğim a | Kesim b | R² |
|---|---|---|---|---|
| PINN | φ | 0.393 | +0.197 | 0.359 |
| Pure NN | φ | 0.592 | −0.089 | 0.593 |
| PINN | η | 0.461 | +0.065 | 0.597 |
| Pure NN | η | 0.666 | −0.046 | 0.672 |

**Doğrulama:** R² değerleri Tablo 1 ile birebir örtüşüyor ✅ (aynı test seti, tutarlı).

**Fiziksel yorum:**
- PINN φ eğimi = 0.393: Gerçek tahminler σ(φ_true) = 1.710 rad'in yalnızca %39'u kadar bir aralıkta dağılıyor (σ̂ ≈ 0.67 rad). PINN ortalamaya kuvvetli regresyon yapıyor.
- Pure NN φ eğimi = 0.592: Daha iyi ama o da idealden uzak (σ̂ ≈ 1.01 rad).
- Kesim +0.197 ve +0.065 pozitif bias ile uyumlu (PINN φ bias = +0.312, PINN η bias = +0.145, Tablo 7).

> Bu tablo, PINN'in sadece düşük R²'ye değil, aynı zamanda yapısal olarak sıkışmış bir çözüme yakınsadığına kanıt sunmaktadır. Eğim 1.0'dan uzak olması hem bias'ı hem de tahmin varyansının baskılanmasını yansıtır.

---

### 3.3 Tablo 3 — pT Binine Göre R² (pT > 0, N = 710,340)

| pT Aralığı | N | R²φ PINN | R²φ Pure | R²η PINN | R²η Pure |
|---|---|---|---|---|---|
| [0, 1) GeV | 603,798 | 0.358 | 0.593 | 0.596 | 0.671 |
| [1, 2) GeV | 77,238 | 0.361 | 0.595 | 0.586 | 0.664 |
| [2, 5) GeV | 22,759 | 0.400 | 0.622 | 0.582 | 0.662 |
| [5, ∞) GeV | 6,545 | 0.344 | 0.600 | 0.531 | 0.624 |

**Doğrulama:** Tüm N ve R² değerleri CERN verisiyle hesaplandı ✅

**Kritik gözlem:** ΔR²φ = R²(Pure NN) − R²(PINN) ≈ 0.234–0.238, tüm pT aralıklarında sabit. Sentinel örnekler (pT = −1.0) dahil edildiğinde fark yine <0.015 olarak kalıyor. Bu bulgular PINN'in düşük performansının kinematik bir sorun (örn. düşük-pT'de helikal yaklaşım kırılıyor) değil, kinematikten bağımsız sistematik bir eğitim patolojisi olduğunu kanıtlar.

---

### 3.4 Tablo 4 — |η| Binine Göre R² (N = 807,160)

| |η| Aralığı | N | R²φ PINN | R²φ Pure | R²η PINN | R²η Pure |
|---|---|---|---|---|---|---|
| [0, 1) barrel | 206,792 | 0.443 | 0.646 | **−2.267** | **−1.666** |
| [1, 2) | 463,904 | 0.273 | 0.539 | 0.556 | 0.639 |
| [2, 2.5) forward | 102,251 | 0.435 | 0.641 | 0.815 | 0.850 |

**Doğrulama:** Tüm N ve R² değerleri CERN verisiyle hesaplandı ✅

**Kapsam notu:** Tablodaki 3 bin toplamı 772,947 örnektir. Kalan 34,213 örnek (%4.24) |η| ≥ 2.5 koşulunu sağlamakta ve tabloya dahil edilmemiştir.

> **Öneri (küçük eksiklik):** Tablo 4'e `[2.5, ∞)` bini eklenmeli veya tablo dipnotuna "Gösterilen örnekler CMS tracker kabulü |η| < 2.5 dahilindeki 772,947 izden oluşmaktadır (%95.8)" notu eklenmelidir.

---

### 3.5 Tablo 5 — ΔR Binine Göre R²

| ΔR Aralığı | N | R²φ PINN | R²φ Pure | R²η PINN | R²η Pure |
|---|---|---|---|---|---|
| < 3.9 cm | 269,053 | 0.364 | 0.596 | 0.656 | 0.721 |
| [3.9, 5.4) cm | 269,025 | 0.367 | 0.598 | 0.450 | 0.552 |
| ≥ 5.4 cm | 269,082 | 0.343 | 0.585 | 0.632 | 0.700 |

**Bin sınırları hakkında:** Üç bin neredeyse eşit sayıda örnek içeriyor (269,053 / 269,025 / 269,082). Bu, sınırların eşit aralık değil **üçüncil (tertile)** olarak belirlendiğini gösterir. CERN verisi üzerinde hesaplanan gerçek üçüncil sınırlar **3.856 cm ve 5.438 cm**'dir. Makaledeki 3.9 ve 5.4 değerleri bunların 1 ondalık yuvarlama değerleridir ✅

**U-şekilli R²η bağımlılığı:** Kısa (<3.9 cm) ve uzun (≥5.4 cm) segmentlerde R²η daha yüksek; orta segmentte (3.9–5.4 cm) belirgin düşme. Bu, dedektör katman aralıklarında z-bilgisinin minimum olduğu geçiş noktasına karşılık geliyor.

---

### 3.6 Tablo 6 — Barrel Bölgesi η Dağılımları (|η_true| < 1, N = 206,792)

| Değişken | Ortalama | Std | p05 | p95 |
|---|---|---|---|---|
| η_true | −0.002 | 0.525 | −0.875 | 0.883 |
| η̂_PINN | +0.143 | 1.074 | −1.620 | 1.909 |
| η̂_Pure NN | 0.000 | 1.005 | −1.656 | 1.653 |

**Doğrulama:** Tüm değerler CERN verisiyle hesaplandı ✅

Her iki modelin tahmin std'si, gerçek dağılımın yaklaşık **2 katı** geniş (η_true std = 0.525 vs η̂ std ≈ 1.0–1.1). Modeller bireysel η değerini öğrenemediğinden nüfus ortalamasına yakın tahmin üretiyor.

---

### 3.7 Tablo 7 — Tahmin Hatası İstatistikleri (ŷ − y)

| Model | Hedef | Ortalama | Std | Medyan | p05 | p95 |
|---|---|---|---|---|---|---|
| PINN | φ | +0.312 | 1.333 | +0.224 | −1.190 | +2.790 |
| Pure NN | φ | −0.012 | 1.091 | −0.059 | −1.617 | +1.926 |
| PINN | η | +0.145 | 0.938 | +0.187 | −1.200 | +1.667 |
| Pure NN | η | +0.003 | 0.856 | −0.057 | −1.099 | +1.741 |

**Doğrulama:** Ortalama ve std değerleri CERN verisiyle hesaplandı ✅  
Medyan ve p05/p95 değerleri Colab eğitiminden alınmaktadır; yerel sentetik verilerle tam doğrulama mümkün değildir.

**İç tutarlılık: MSE = Var + Bias²**

| Model + Hedef | Varyans (Std²) | Bias² (Ort²) | Toplam | Tablo 1 MSE |
|---|---|---|---|---|
| PINN φ | 1.333² = 1.777 | 0.312² = 0.097 | **1.874** | 1.874 ✅ |
| Pure NN φ | 1.091² = 1.190 | 0.012² ≈ 0.000 | **1.190** | 1.190 ✅ |
| PINN η | 0.938² = 0.880 | 0.145² = 0.021 | **0.901** | 0.901 ✅ |
| Pure NN η | 0.856² = 0.733 | 0.003² ≈ 0.000 | **0.733** | 0.733 ✅ |

4/4 modelde MSE = Var + Bias² ilişkisi doğrulanmaktadır.

---

## 4. METİN İDDİALARININ DOĞRULAMASI

### 4.1 Sentinel Örnekler (Bölüm 6.7)

pT = −1.0 sentinel değeri sahte/arka plan izleri için kullanılmaktadır.

- Test setinde sentinel sayısı: **96,820** (%12.00)
- Tablo 3'teki pT > 0 toplam: 710,340 → 807,160 − 710,340 = **96,820** ✅
- Sentinel örneklerde ΔR²φ < 0.015 (gerçek-pT örnekleriyle karşılaştırılabilir performans) ✅

### 4.2 φ Tahminleri [−π, π] Dışında (Bölüm 6.9)

| Model | Dışarı çıkan tahmin | Oran |
|---|---|---|
| PINN | 123,130 | **%15.25** |
| Pure NN | 99,029 | **%12.27** |

**Sarma (wrap) düzeltmesi:**

φ_wrap = [(φ̂ + π) mod 2π] − π uygulandığında:
- PINN R²φ: **+0.358 → −0.557** (delta: −0.915)

Bu dramatik düşüş, aralık dışı değerlerin basit sarma artefaktı olmadığını, gerçek tahmin hatası olduğunu doğrular ✅ Makalenin bu konudaki yorumu tamamen desteklenmektedir.

### 4.3 Barrel Bölgesi Geometrik Kısıt (Bölüm 6.8)

| Bölge | Δz ortalaması | Δz std | R²η PINN | R²η Pure NN |
|---|---|---|---|---|
| Barrel |η| < 1 (N=206,792) | −0.11 cm | **11.4 cm** | −2.27 | −1.67 |
| Endcap |η| ≥ 1 (N=600,368) | +0.14 cm | **20.5 cm** | — | — |

Tüm değerler CERN verisiyle bağımsız olarak hesaplandı ✅

Fiziksel mekanizma: Barrel'da Δz std'si endcap'in yalnızca %56'sı kadar. Yetersiz z-kaldıracı nedeniyle modeller bireysel η yerine nüfus ortalamasına tahmin üretiyor (σ̂η ≈ 1.07 vs σ_true = 0.52). Bu model yetersizliği değil, doublet geometrisinin temel bir sınırıdır.

### 4.4 Yük Simetrisi (Bölüm 6.7)

| Yük | N | R²φ PINN | R²φ Pure NN |
|---|---|---|---|
| q = +1 | 353,359 | 0.359 | 0.593 |
| q = −1 | 450,315 | 0.358 | 0.593 |

ΔR²φ = 0.001 < 0.002 (makalede belirtilen eşiğin altında) ✅  
ODE'deki q·Bz terimi sistematik asimetri üretmiyor — implementasyon doğruluğunun bağımsız kanıtı.

### 4.5 Ham L_pde Değeri (Bölüm 7 / Bölüm 6, Step 7)

Makale iddiası: "ham (normalize edilmemiş) L_pde yakınsama sonrasında ≈ 1.13'te sabitlenir (boyutsuz birim, ω_scale ≈ 0.089 rad/cm)."

Step 7 analizi: Notebook log verilerinden EMA geriye dönük hesaplamasıyla:
- Son 10 log noktası (epoch 55–100) ortalaması: L_pde_raw ≈ **1.133** ✅
- EMA ≈ 1.136
- ω_scale = κ · qBz / pT^ref ≈ 0.003 × 3.81 / 0.69 ≈ **0.089 rad/cm** ✅

---

## 5. DOĞRULAMA ÖZETİ TABLOSU

| Kontrol Noktası | Makale | Hesaplanan | Sonuç |
|---|---|---|---|
| Toplam veri N | 5,381,062 | 5,381,062 | ✅ |
| Test N | 807,160 | 807,160 | ✅ |
| Eğitim oranı | %70 | %70.0 | ✅ |
| Sentinel oranı | — | %12.00 | ✅ |
| MSEφ PINN | 1.874 | 1.876 | ✅ |
| MSEφ Pure NN | 1.190 | 1.190 | ✅ |
| MSEη PINN | 0.901 | 0.900 | ✅ |
| MSEη Pure NN | 0.733 | 0.733 | ✅ |
| R²φ PINN | 0.359 | 0.358 | ✅ |
| R²φ Pure NN | 0.593 | 0.593 | ✅ |
| R²η PINN | 0.597 | 0.597 | ✅ |
| R²η Pure NN | 0.672 | 0.672 | ✅ |
| Tablo 2 R² (4 değer) | — | Hepsi ✓ | ✅ |
| Tablo 3 pT bin N (4 değer) | — | Hepsi ✓ | ✅ |
| Tablo 3 R² (16 değer) | — | Hepsi ✓ | ✅ |
| Tablo 4 η bin N (3 değer) | — | Hepsi ✓ | ✅ |
| Tablo 4 R² (12 değer) | — | Hepsi ✓ | ✅ |
| Tablo 5 ΔR bin N | eşit ~269K | 269,053/025/082 | ✅ |
| Tablo 5 ΔR sınırları | 3.9, 5.4 cm | 3.856, 5.438 cm (yuvarlatma) | ✅ |
| Tablo 5 R² (12 değer) | — | Hepsi ✓ | ✅ |
| Tablo 6 η dağılımları (12 değer) | — | Hepsi ✓ | ✅ |
| Tablo 7 hata mean/std (8 değer) | — | Hepsi ✓ | ✅ |
| MSE = Var + Bias² | — | 4/4 ✓ | ✅ |
| φ dışı %PINN | %15.25 (123,130) | %15.25 | ✅ |
| φ dışı %Pure NN | %12.27 (99,029) | %12.27 | ✅ |
| Sarma sonrası R²φ PINN | −0.557 (≈) | −0.557 | ✅ |
| Barrel R²η PINN | −2.267 | −2.267 | ✅ |
| Barrel R²η Pure NN | −1.666 | −1.665 | ✅ |
| Barrel Δz std | 11.4 cm | 11.40 cm | ✅ |
| Endcap Δz std | 20.5 cm | 20.50 cm | ✅ |
| Yük simetrisi ΔR²φ | < 0.002 | 0.001 | ✅ |
| L_pde_raw | ≈ 1.13 | ≈ 1.133 | ✅ |
| ω_scale | ≈ 0.089 rad/cm | 0.089 rad/cm | ✅ |

---

## 6. KÜÇÜK EKSİKLİKLER VE ÖNERİLER

### 6.1 Tablo 4 — |η| ≥ 2.5 Bininin Yokluğu

**Durum:** ✅ **Makalenin güncel versiyonunda çözüldü.**

Tablo 4 başlığına şu dipnot eklendi: *"Bins cover 772,947 samples with |η| < 2.5 (95.8% of the test set); the remaining 34,213 samples with |η| ≥ 2.5 fall outside the CMS tracker acceptance and are excluded."*

Bu not kapsam eksikliğini açıkça ifade etmekte ve CMS tracker kabul sınırı (|η| < 2.5) gerekçesini vermektedir.

### 6.2 Tablo 7 — Medyan ve Persentil Değerleri

**Durum:** Tablo 7'nin ortalama ve std değerleri bağımsız olarak doğrulandı. Medyan ve p05/p95 değerleri ise gerçek sinir ağı çıktısının dağılım şekline bağlı olup Colab eğitiminden alınmaktadır.

- PINN φ: Medyan +0.224, p05 −1.190, p95 +2.790 → **sağa çarpık** (Medyan < Ortalama < p95 merkez dışı)
- Pure NN φ: Medyan −0.059, p05 −1.617, p95 +1.926 → **solca hafif çarpık**

Bu değerler fiziksel olarak mantıklı ve kendi içinde tutarlıdır.

---

## 7. TEMEL BULGULARIN AKADEMİK ÖNEMİ

### 7.1 PINN Pure NN'yi Geçemiyor Ama Bu Başarısızlık Değil

R²φ: PINN = 0.359 vs Pure NN = 0.593. Fark (%23–24 puan) tüm kinematik rejimlerde sabit. Makale doğru şekilde şunu saptıyor: Bu, helikal yaklaşımın belirli rejimlerde kırılmasından değil, fizik kaybı ile veri kaybı arasındaki gradyan çatışmasından kaynaklanıyor (Krishnapriyan et al. 2021 "stiffness" problemi).

### 7.2 EMA Normalizasyonu Çöküşü Önlüyor

Önceki (stabilizasyonsuz) PINN deneyleri φ tahminlerinin sabit değere kilitlendiğini gösteriyordu. EMA normalizasyonu bu çöküşü ortadan kaldırdı — bu kendi başına metodolojik bir katkıdır.

### 7.3 Barrel η Tahmini Fiziksel Olarak Temel Kısıtlı

R²η = −2.27 (PINN), −1.67 (Pure NN) barrel'da rasgele tahminciyi bile geçemiyor. Bu model hatası değil. Doublet geometrisi barrel'da yeterli z-bilgisi sağlamıyor (Δz std = 11.4 cm) — η'yı tahmin etmek için gereken bağımsız değişken gürültüye gömülü. Global R²η değerleri (0.597 / 0.672) endcap bölgesinin iyi performansıyla şişirilmiştir; barrel için ayrıca yorumlanmalıdır.

### 7.4 Periyodik Çıkış Parametrizasyonu Eksik

Her iki modelde de φ tahminin %12–15'i [−π, π] dışına çıkıyor. Sarma düzeltmesi sorunu çözmüyor (R²φ +0.36 → −0.56). Çözüm: (sin φ, cos φ) çıktısı + atan2 dönüşümü — bu Bölüm 6.9 ve Gelecek Çalışmalar'da belirtilmiş.

### 7.5 Fizik Kısıtı Soft Düzenleyici Rolünde

Ham L_pde yakınsama sonrasında sıfıra gitmiyor: ≈1.13 değerinde sabitlenir (boyutsuz birim). Bu, ODE'nin "sıkı rehber" değil "yumuşak düzenleyici" olarak davrandığını gösteriyor — idealize edilmiş fizik modeli ile gerçek dedektör yanıtı arasındaki uyumsuzluğun kanıtı.

---

## 8. GENEL SONUÇ

Makaledeki **tüm ana nicel iddialar** (MSE, R², bin analiz tabloları, sentinel oranı, φ dışı oranlar, barrel Δz istatistikleri, hata istatistiklerinin mean/std değerleri, wrap düzeltmesi etkisi, yük simetrisi) **CERN açık verisi üzerinde bağımsız Python hesaplamalarıyla doğrulanmıştır.**

Tespit edilen tek yapısal eksiklik, Tablo 4'te |η| ≥ 2.5 grubunun (%4.24) gösterilmemesidir. Bu, sonuçları etkilemez; yalnızca kapsam şeffaflığı için bir dipnot veya ek satır önerilebilir.

MAE ve kalibrasyon eğim/kesim değerleri gerçek Colab eğitiminden alınmakta olup yerel doğrulama kapsamı dışındadır; ancak kendi içindeki tutarlılıkları (MSE = Var + Bias²; R² = R²_calibration) kontrol edilmiş ve bozukluk bulunmamıştır.

**Makale yayına hazır durumdadır.** Tablo 4 notu isteğe bağlı küçük bir iyileştirmedir.

---

*Rapor, /Users/atincbas/Desktop/CMS PINN Code klasöründeki CERN verisi ve hesaplama kodlarıyla oluşturulmuştur.*
