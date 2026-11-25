# NeuroTune: Sonraki Adımlar ve Deney Planı

**Tarih:** 17 Kasım 2026  
**Durum:** P0 validation-only A/B/C/D ablation tamamlandı; yeni bağımsız test kohortu P1 olarak bekliyor  
**Kaynak:** `NeuroTune_durum_raporu_2026-11-16.md` (16 Kasım'da oluşturulmuş, 17 Kasım'da P0 sonuçlarıyla güncellenmiş durum raporu)

## 1. Amaç ve karar

NeuroTune, şarkıların **zamansal akustik özelliklerini (pitch, timbre, loudness, segment timing)** merkezine alan; **BPM, genre ve lyrics** özelliklerini de kullanarak ortak şarkı embedding'leri öğrenen müzik benzerliği/retrieval projesidir.

Bu aşamadaki karar:

1. **Mevcut C modelini ve sonuçlarını dondur.** Görülmüş test sonuçları üzerinde yeniden model/hiperparametre seçimi yapma.
2. **Mevcut D modelini doğrula:** time-series audio + BPM + genre + lyrics encoder'ı kodda zaten mevcut.
3. **Adil ablation karşılaştırmaları yap:** yalnızca audio, audio + BPM, audio + BPM + genre, tam multimodal model.
4. **Yeni bağımsız artist-held-out test kohortu oluştur.** Nihai performans iddiasını daha önce incelenmiş test metriklerine dayandırma.
5. **Retrieval demosunu** eğitim ve değerlendirme tamamlandıktan sonra ekle.

**Öncelik:** Yeni MSD arşivleri indirmeden önce geçerli bir deney protokolünün tamamlanması. Daha çok şarkı tek başına daha güvenilir ölçüm anlamına gelmez.

## 2. Raporla doğrulanan mevcut durum

13.822 track ve önceki C sonuçları mevcut durum raporundan alınmıştır; P0 validation sonuçları 17 Kasım'da `results/neurotune_v1_full_modal_ablation/experiment.json` üzerinden kaydedilmiştir.

| Bileşen | Mevcut durum |
| --- | --- |
| Veri | MSD A–D pilotunda **13.822** kullanılabilir tam-modlu parça |
| Akustik girdi | İlk 30 saniyeden **T × 26**, değişken uzunluklu, ham/normalizasyon uygulanmamış segment dizileri |
| Bağlantılar | **67.657** Last.fm train-source ve **8.033** resmî test-source kenarı (katalog düzeyinde) |
| Güncel deney | **P0 A/B/C/D validation-only ablation**; A/B/D yeni eğitildi, C read-only validation referansı |
| Audio encoder | Masked `26 → 32 → 64 → 64` Conv1D + masked temporal mean pooling |
| Auxiliary encoder | BPM `1 → 8`, genre `G → 16`, lyrics `5000 → 32` |
| Eğitim | Cosine triplet-margin loss, margin `0.2`, Adam `0.001`, batch `64`, CPU, `50` epoch |
| Değerlendirme | **3 artist-grouped fold × 3 seed**, yalnız train/validation özellikleri ve etiketleri; mevcut test cohort'u kullanılmadı |
| P0 çıktıları | 36 model validation kaydı, 12 matched-cosine kaydı, 27 yeni checkpoint, 9 schedule kopyası, 1.800 curve satırı |
| Doğrulama | P0 sonrası 423 test geçti; compileall ve `pip check` başarılı. Artifact/checkpoint provenance kontrolleri de geçti. |

> **Boyut doğrulama notu:** C'de audio/BPM/genre çıktıları sırasıyla 64/8/16 boyutludur; bu nedenle C fusion `88 → 128 → 64` olur. D'de lyrics kolu da mevcuttur ve 32D üretir; 64 + 8 + 16 + 32 = 120 olduğundan D fusion `120 → 128 → 64` olur.

> **Kod denetimi kararı:** `NeuroTuneV1Encoder` içinde D varyantı ve lyrics kolu zaten uygulanmıştır; D encoder'ını ekleme işi tamamlanmış durumdadır. Mevcut A–D test kohortunda değerlendirme, yeni bağımsız bir kohort oluşturulana kadar ertelenir.

### 2.1 Önceki C modelinin test sonuçları (tarihsel)

Değerler mevcut rapordaki ortalama ± popülasyon standart sapmasıdır. C satırları **9 fold-seed koşusu**, cosine baseline satırları **3 fold ölçümü** içeriyor; bu gözlemler bağımsız 9 test kohortu olarak yorumlanmamalıdır.

| Etiket kaynağı | Model | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Train-source'dan ayrılan test kenarları | C | 0.1681 ± 0.0274 | 0.2232 ± 0.0347 | 0.0769 ± 0.0186 | 0.0895 ± 0.0178 |
| Train-source'dan ayrılan test kenarları | Matched cosine | 0.1510 ± 0.0248 | 0.2022 ± 0.0309 | 0.0682 ± 0.0089 | 0.0787 ± 0.0126 |
| Resmî Last.fm test-source | C | 0.1521 ± 0.0556 | 0.1802 ± 0.0699 | 0.0560 ± 0.0252 | 0.0760 ± 0.0292 |
| Resmî Last.fm test-source | Matched cosine | 0.1649 ± 0.0716 | 0.1970 ± 0.0854 | 0.0801 ± 0.0449 | 0.0920 ± 0.0446 |

**Yorum:** C, train-source'dan ayrılan bağlantılarda cosine baseline'ın önünde; resmî test-source'da dört metriğin tamamında gerisinde. Resmî test tarafında fold başına yalnızca **30–50 pozitif kenar** var. Bu, C'nin genel üstünlüğünü göstermek için yeterli değil.

### 2.2 P0 full-modal ablation — validation-only

13.822 track'in tamamında musiXmatch kaydı bulundu; boş lyric dokümanı yoktu. A/B/D yeni eğitildi, dokuz C checkpoint'i read-only kullanıldı. Bu çalışma **mevcut test cohort'una özellik, embedding, relevance veya metrik üretmedi**. A/B/D ve cosine karşılaştırmaları aynı üç fold'un validation galerileri/kenarları üzerindedir; checkpoint seçimi de validation NDCG@10 ile yapıldığı için sonuçlar yalnızca geliştirme verisidir.

Model ortalamaları 9 fold-seed; cosine ortalamaları varyant başına 3 fold gözlemidir:

| Variant | Model Recall / HitRate / MRR / NDCG | Matched cosine Recall / HitRate / MRR / NDCG |
| --- | --- | --- |
| A — audio | 0.048795 / 0.088177 / 0.034365 / 0.030221 | 0.041842 / 0.075427 / 0.031908 / 0.027089 |
| B — audio+BPM | 0.043948 / 0.078437 / 0.028857 / 0.026365 | 0.041812 / 0.080982 / 0.035022 / 0.028011 |
| C — audio+BPM+genre (read-only) | 0.118594 / 0.193172 / 0.069910 / 0.068638 | 0.109255 / 0.190992 / 0.078063 / 0.068781 |
| D — audio+BPM+genre+lyrics | 0.119240 / 0.194019 / 0.072913 / 0.071530 | 0.121607 / 0.204624 / 0.087984 / 0.077921 |

**Yorum:** D learned model C'den dört metrikte de az farkla yüksek, fakat D matched cosine baseline dört metriğin tamamında D modelinden yüksek. C modeli Recall@10 ve HitRate@10'da cosine'ı geçerken MRR@10 ve NDCG@10'da az farkla geride. A/B daha düşük sonuç verdi. Bu ölçümler validation ile model seçimi yapıldığından bağımsız performans tahmini değildir; Last.fm'de gözlenmeyen çiftler doğrulanmış negatif sayılmaz. Hiçbir genelleme veya model üstünlüğü iddiası yapılmamalı.

### 2.3 Değerlendirme sınırlılıkları

- Daha önce incelenmiş A/B ve A–D test sonuçları **yeni geliştirme kararları için kullanılmamalı**.
- Farklı deneylerin katalog, split ve relevance map'leri farklı olabilir; sonuçları doğrudan yan yana koyup üstünlük iddiası kurma.
- Last.fm'de kaydı bulunmayan bir çift **kanıtlanmış negatif değildir**.
- Aynı etiket kaynak havuzundan gelen fold'lar ve seed'ler bağımsız örnekler değildir.
- Lyrics eklemek modelin daha iyi olacağını garanti etmez; amaç bunu ölçmektir.

## 3. Aşama 1: Mevcut C deneyini dondurma

**Yapılacaklar**

- `results/neurotune_v1_abcd_cv/experiment.json`, `training_curves.csv`, checkpoint'ler ve triplet schedule'larını değiştirme.
- Split tanımlarını, artist listelerini, katalog kimliklerini, kullanılan bilinen pozitif bağlantı kümelerini, kaynak dosyası hash'lerini ve preprocessing konfigürasyonunu manifest içinde koru.
- C modeli sonuçlarını **exploratory / previously inspected test results** olarak işaretle.
- Model veya loss seçimi için yalnızca mevcut **train/validation** politikası kullanılabilir; görülmüş test skoru optimize edilemez.
- GPU/CPU, paket sürümleri ve random seed'leri kayıt altına al.

**Kabul kriteri:** D modeli geliştirilmeden önce mevcut C çıktılarının değişmediği doğrulanıyor; yeni çalışmanın dosyaları ayrı deney namespace'ine yazılıyor.

## 4. Aşama 2: Mevcut D lyrics encoder'ını doğrulama — tamamlandı

P0 kapsamında D encoder'ı, train-only lyrics dönüşümü ve eksik/boş doküman politikası doğrulandı. A–D kataloğunun tamamında MXM kaydı vardı; sıfır boş doküman raporlandı. Validation-only sonuçları bölüm 2.2'dedir.

### 4.1 Nihai hedef mimari

```text
Audio T×26 (ilk 30 s, ragged)
    └─ masked Conv1D × 3 + masked temporal pooling ──> 64D
BPM (1 değer)
    └─ küçük MLP ─────────────────────────────────────>  8D
Genre (one-hot / uygun kodlama)
    └─ küçük MLP ─────────────────────────────────────> 16D
Lyrics (musiXmatch BoW/TF-IDF)
    └─ küçük MLP ─────────────────────────────────────> 32D

Concat (64+8+16+32 = 120D)
    └─ Fusion MLP 120 → 128 → 64
    └─ L2 normalization
    └─ 64D song embedding

Training: shared-weight Siamese/triplet encoder
Retrieval: cosine similarity → Top-K
```

**Not:** Bu D mimarisi mevcut `NeuroTuneV1Encoder` içinde uygulanmıştır; C fusion girişi `88`, D fusion girişi `120` boyutludur.

### 4.2 Lyrics hazırlama

- musiXmatch verisini **tam şarkı sözleri** olarak değil, lisans ve veri biçimine uygun **bag-of-words / kelime sayımı** olarak işle.
- Lyrics kaydını güvenilir `track_id` eşleşmesiyle şarkıya bağla; başlık/sanatçı adına göre sessiz fuzzy matching yapma.
- TF-IDF veya başka bir veri üzerinden öğrenilen dönüşüm kullanılıyorsa **yalnızca train fold üzerinde fit et**, validation/test'e `transform` uygula.
- Seyrek kelime verisini mümkün olduğunca sparse sakla; gerekirse encoder girişinde kontrollü yoğunlaştır.
- Eksik lyrics için açık bir politika tanımla: tam-modlu kesişim veya explicit mask; metin bulunmayan şarkılara rastgele değer atama.
- Üç modaliteli C ile dört modaliteli D'yi **aynı eligible şarkı, artist partition, query ve gallery** üzerinde karşılaştır. Gerekirse C'yi ortak alt kümede yeniden eğit; ama bunu yeni ve validation ile belirlenmiş protokolle yap.

### 4.3 Eğitim ve teknik kurallar

- Mevcut triplet-margin yaklaşımını başlangıç olarak koru: cosine distance, margin `0.2`.
- Bilinen pozitiflerden negatif seçme; diğer gözlenmemiş çiftlerin kesin negatif olmadığı sınırlılığını kaydet.
- Checkpoint seçimi yalnızca validation NDCG@10 ile yapılsın; eşitlik kuralları önceden sabitlensin.
- Artist-grouped ayrımı ve feature normalization train-only yaklaşımı korunsun.
- Masked convolution/pooling, padding veya segment uzunluğundan kaynaklı bilgi sızıntısını önlemeli.
- Lyrics kolunu eklemek için yeni parametrelerin sayısını ve bellek/hesaplama maliyetini raporla.

**Kabul kriteri:** D modeli finite 64D embedding üretiyor; gradients lyrics encoder'a ulaşıyor; aynı parçaya deterministik inference yapılıyor; train-only transforms ve negatif örnek testleri geçiyor.

## 5. Aşama 3: Ablation study — P0 tamamlandı

Önceden belirlenen A/B/C/D ablation, mevcut test cohort'u açılmadan train/validation üzerinde yürütüldü. Ölçülen metrikler ve sınırlılıklar bölüm 2.2'de; tüm kaynak, checkpoint ve schedule ayrıntıları `results/neurotune_v1_full_modal_ablation/experiment.json` dosyasındadır.

Bütün karşılaştırmaları **aynı şarkı kataloğu, artist split'leri, aday gallery, relevance tanımı ve mümkün olduğunca aynı eğitim bütçesiyle** yap.

| Deney | Time-series audio | BPM | Genre | Lyrics | Amaç |
| --- | :---: | :---: | :---: | :---: | --- |
| A | ✓ |  |  |  | Ses zaman serisi tek başına ne kadar güçlü? |
| B | ✓ | ✓ |  |  | Tempo katkısı var mı? |
| C | ✓ | ✓ | ✓ |  | Genre katkısı var mı? |
| D | ✓ | ✓ | ✓ | ✓ | Lyrics eklenmesi retrieval'ı iyileştiriyor mu? |
| Baseline | Sabit, el yapımı/mean-pooled özellikler | Uygunsa | Uygunsa | Uygunsa | Öğrenilmiş embedding gerekli mi? |

**Deney tasarımı**

- Her varyant için en az 3 seed; compute uygunsa mevcut 3 artist fold düzenini koru.
- A/B/C/D karşılaştırmasında test şarkılarını ve gallery'yi değiştirme; her satırda aynı test sorularını kullan.
- Her model için `Recall@10`, `HitRate@10`, `MRR@10`, `NDCG@10` raporla; ikinci hedef olarak training/inference maliyetini göster.
- Sadece model ortalaması değil, **fold ve seed ayrıntılarını** da sakla.
- Fold/artist kümeleri üstünden belirsizlik raporu üret; 9 fold-seed koşusunu 9 bağımsız veri kümesi gibi istatistiksel test etme.
- Görülen sonuçlar beklentiyi karşılamasa dahi geriye dönük seçici raporlama yapma.

**Önemli kural:** Önceden görülmüş test setinde yeni ablation tasarlayıp varyantı test skoruna göre seçmek final üstünlük iddiasını geçersizleştirir. Model ve deney tasarımı train/validation'da belirlenmeli, nihai doğrulama **yeni bağımsız test** ile yapılmalı.

## 6. Aşama 4: Yeni bağımsız test kohortu

Mevcut A–D ve resmî 10K test sonuçları daha önce incelendi. Bunları final performans iddiasında yeni, bakılmamış holdout gibi kullanma.

**Hedef özellikler**

1. Model seçimi bittikten sonra açılacak, önceden ayrılmış **artist-disjoint** şarkılar.
2. Eğitim sırasında modelin gördüğü label ilişkileriyle çakışmayan test kenarları.
3. Katalog başına yeterli sayıda gözlenmiş pozitif ve sorgu. Şarkı sayısından çok **pozitif kenar / değerlendirilebilir sorgu sayısı** önemlidir.
4. Lisansı, kaynağı, track ID eşleştirmesi ve checksum/provenansı doğrulanmış veri.
5. Bilinen bütün pozitiflerin negative sampling'den dışlanması; bazı bilinmeyen pozitiflerin yine de kaçırılmış olabileceğinin belgelenmesi.
6. Query/gallery tanımının, unseen-artist kısıtının, relevance kriterinin ve model seçimini sonlandıran tarihin deney başlamadan yazılması.

**Önerilen sonuç raporu**

- Kaynak bazında ayır: Last.fm train-source'dan ayrılan ilişkiler ve resmî test-source ilişkileri aynı veri değildir.
- Her fold için: test artist, şarkı, pozitif kenar, test query ve gallery sayısı.
- Model bazında: Recall@10, HitRate@10, MRR@10, NDCG@10 ve belirsizlik.
- Yeni holdout elde edilemezse: sonuçları **keşifsel** olarak sun, genelleme/üstünlük iddiasında bulunma.

**Veri kaynağı notu:** A–D üçüncü taraf HDF5 arşivlerinin kamuya açık lisansı ve kaynak bütünlüğü mevcut rapora göre bağımsız doğrulanmamış. Yeniden dağıtım, yayımlama veya yeni test kaynağı seçimi öncesinde bu konu çözülmeli.

## 7. Aşama 5: Retrieval demo

Demoyu **değerlendirmeden sonra**, kullanıcı deneyimi katmanı olarak ekle; model başarısının yerine geçmesin.

**Minimum özellikler**

- Katalogdan şarkı seçme / başlık-sanatçı arama.
- Önceden hesaplanmış 64D embedding üzerinden cosine similarity.
- Query şarkısı hariç tutularak Top-10 sonuçları gösterme.
- Başlık, sanatçı, genre, BPM ve benzerlik skorunu sunma.
- Mümkünse A/C/D modelleri arasında kullanıcı tarafından seçilebilen karşılaştırma.
- Eksik lyrics/genre ve telifli ses dosyası bulunmama durumlarını açıkça belirtme.

10K–20K ölçeğinde brute-force vektör karşılaştırması başlangıç için yeterli olabilir; daha büyük katalogda ANN indeks isteğe bağlıdır. Index yapısı değerlendirme sonuçlarını değiştirmemeli veya etkisi ayrıca ölçülmelidir.

## 8. Önceliklendirilmiş teknik görevler

### P0 — Teslim için zorunlu

- [x] Mevcut D lyrics encoder'ını ve modalite maskesi/eksik veri politikasını doğrula.
- [x] C/D fusion giriş boyutlarını kod üzerinde doğrula ve dokümante et: C `88 → 128 → 64`, D `120 → 128 → 64`.
- [x] Train-only lyrics preprocessing, artist split ve negative sampler testlerini ekle.
- [x] Önceden belirlenmiş, ortak kataloglu A/B/C/D validation-only ablation protokolünü çalıştır.
- [x] Sonuçları kaynak/fold/seed bazında yeniden üretilebilir JSON/CSV olarak kaydet.
- [x] Önceki C raporundaki test-source cosine sonucunu tarihsel olarak raporla; P0'da yeniden değerlendirme yapma veya seçime katma.

### P1 — Bilimsel güvenilirliği artırır

- [ ] Görülmemiş, lisansı doğrulanmış yeni artist-held-out test kohortu oluştur.
- [ ] Test sorgusu/pozitif kenar yoğunluğu raporunu ve metrik belirsizliklerini ekle.
- [ ] Lyrics katkısı için `D − C` karşılaştırmasını aynı test örneklerinde incele.
- [ ] Veri kaynaklarının provenansını ve lisansını doğrula.

### P2 — Opsiyonel ürünleştirme

- [ ] Basit Top-10 retrieval arayüzü ekle.
- [ ] Hız, embedding üretim süresi ve bellek ölçümü ekle.
- [ ] Sonuçların açıklanabilirliği için feature-group ablation görselleri oluştur.

## 9. Agent uygulaması için çalışma kuralları

1. **Mevcut deneyleri ve sonuç klasörlerini değiştirme.** Yeni iş için örneğin `results/neurotune_v1_full_modal_ablation/` altında ayrı namespace aç.
2. Önce veri protokolünü ve kabul kriterlerini yaz; sonra model değişikliklerini yap.
3. **Eski test skorunu hiperparametre optimizasyonunda kullanma.** Train/validation dışında skor bakarak karar verme.
4. Raporlarda *measured*, *reported*, *planned* etiketlerini ayır; planlanan deneyi çalıştırılmış gibi yazma.
5. Ablation karşılaştırmalarında aynı değerlendirme evrenini koru; aksi durumda farkı mimariye atfetme.
6. Kullanılan kod sürümü, yapılandırma, seed, veri ve transform hash'lerini sakla.
7. Testler ve ölçümler gerçekten çalışmadan “başarılı”, “doğrulandı”, “üstün” ifadeleri kullanma.
8. CMP684 ders teslim formatını ayrıca doğrula; Python kullanımı proje için daha önce uygun görülmüş olsa da rapor ve ek dosya koşullarını kontrol et.

## 10. Önerilen final rapor iskeleti

1. **Introduction:** Müzik benzerliği problemi, time-series + metadata ihtiyacı, araştırma sorusu.
2. **Related Work:** Music similarity, multimodal audio encoder, metric learning.
3. **Dataset & Protocol:** MSD A–D, musiXmatch, tagtraum, Last.fm, eşleşmeler, artist-held-out split, veri kaynakları/lisans.
4. **Method:** Temporal Conv1D, BPM/genre/lyrics encoders, fusion, 64D embedding, triplet objective.
5. **Experiments:** Baseline, A/B/C/D ablation, hyperparameters, seed'ler, retrieval metrikleri.
6. **Results:** Başarılar ve başarısızlıklar, fold farkları, test kaynakları, uncertainty.
7. **Discussion & Limitations:** Seyrek pozitif etiketler, unknown negatives, metadata bias, data provenance, test reuse.
8. **Conclusion:** Hangi modalitelerin hangi koşullarda katkı verdiğine ilişkin kanıtla sınırlı sonuç.

## 11. Tamamlanma kriterleri

- **Teknik:** D modelinin inference ve eğitim testleri geçiyor; embedding'ler finite ve 64D; veri split/preprocessing sızıntısı denetimleri başarılı.
- **Bilimsel:** A/B/C/D ve baseline aynı protokolde ölçülüyor; değerlendirme sonuçları model seçimine sızmıyor; belirsizlik ve etiket kaynağı ayrımı korunuyor.
- **Kaynak bütünlüğü:** Veri/deney hash'leri, seed'ler, parametreler, checkpoint ve sonuç tabloları kayıtlı; lisans/provenans belirsizliği açıkça raporlanıyor.
- **Teslim:** Raporun iddiaları kaydedilmiş deneyleri aşmıyor; kod ve yeniden üretim komutları çalışır durumda; demo varsa ek bir bileşen olarak sunuluyor.

---

**Kısa sonuç:** P0 full-modal A/B/C/D validation-only ablation tamamlandı. Sonraki bilimsel adım P1 olarak lisansı/provenansı doğrulanmış, daha önce kullanılmamış artist-held-out test kohortu hazırlamaktır. Eski test cohort'u yeni bağımsız kohort oluşana kadar değerlendirilmemeli; mevcut P0 skorları validation seçimine tabi geliştirme sonuçları olarak kalmalıdır.
