# NeuroTune D4 Sonrası: Teşhis, Performans İyileştirme ve Değerlendirme Planı

**Hazırlanma:** 23 Kasım 2026; **son güncelleme:** 24 Kasım 2026  
**Durum:** P0-A, P0-B, P1-A ve P1-B yeniden eğitim yapılmadan tamamlandı. P1-B, 24 Kasım 2026'da salt-okunur çalıştırıldı. P2 protokolü isteğe bağlı gelecek değerlendirme kapısı olarak kaydedildi; kullanıcı bunun ticari ürün değil ders ödevi olduğunu ve daha fazla kaynak/lisans araştırmasına gerek olmadığını belirtti. Bu çalışma için P2 kaynak araştırması ve bağımsız test ertelendi; hiçbir yeni veri/etiket açılmadı. P0 sonuçları `results/neurotune_v1_d4_modality_sensitivity/README.md` ve `results/neurotune_v1_d4_triplet_diagnostics/analysis.md`; P1-A ve P1-B sonuçları sırasıyla `results/neurotune_v1_p1a_intra_genre_retrieval/analysis.md` ve `results/neurotune_v1_p1b_audio_branch_diagnostic/analysis.md` içindedir.
**Birincil kaynak:** `NeuroTune_D4_Son_Testler_2026-11-23.md`; P1-A: `results/neurotune_v1_p1a_intra_genre_retrieval/experiment.json`; P1-B: `results/neurotune_v1_p1b_audio_branch_diagnostic/experiment.json`  
**İkincil kaynaklar:** `NeuroTune_D_Modalite_Duyarliligi_2026-11-17.md`, `NeuroTune_D_Modeli_Uygulama_ve_Validation_Sonuclari_2026-11-17.md`  
**Hedef:** Mevcut D4 modelinin nasıl karar verdiğini açıklamak; modelin audio zaman serisini gerçekten daha etkili kullanıp kullanmadığını ölçmek; en düşük ilave eğitim maliyetiyle yeni araştırma kararını almak.

---

## 1. Karar özeti

1. **D4 modelini mevcut geliştirme referansı olarak koru.** Yeni mimari veya hiperparametre aramasına hemen başlama.
2. **P0 tamamlandı (23 Kasım 2026):** Yeni eğitim yapmadan D4 modalite duyarlılığı ve triplet/embedding teşhisleri çalıştırıldı; D0 ile eşleştirilmiş ve aynı validation protokolünde karşılaştırıldı. Sonuçlar bu belgenin durum bölümünde bağlantılanan artefaktlarda yer alıyor.
3. **P1-A ve P1-B tamamlandı (24 Kasım 2026'ya kadar, yeniden eğitim yok).** Same-genre P1-A'da D4 neural, D0 neural'den daha yüksek macro-query NDCG aldı ama D4-matched cosine'ın gerisinde kaldı. P1-B'de D4 audio-branch tanısı D0 audio branch'ini 9 eşleşmenin 8'inde geçti; yine de D4 neural ve D4-matched cosine'dan düşük kaldı. Bulgular validation-only ve tanısaldır; ürün üstünlüğü/genelleme kanıtı değildir. Ders ödevi kapsamında mevcut sonuçlar yeterlidir; yeni uzun eğitim veya P2 kaynak araştırması başlatma.
4. **P2 isteğe bağlı gelecek aşama olarak ertelendi.** Bağımsız artist-disjoint test ve kabul eşiği taslağı saklandı; ancak kullanıcı bu çalışmanın ticari ürün değil ders ödevi olduğunu belirtti. Daha fazla kaynak/lisans araştırması veya bağımsız test, kullanıcı açıkça istemedikçe yapılmayacak.
5. D1–D3'ün yalnızca artefakt bütünlüğü bu son D4 raporunda doğrulanıyor. **D1–D3 performans sıralamasını bu belgeden türetme**; kendi raporlarından ayrı çıkar.

**Temel araştırma sorusu:** *Öğrenilmiş çok-modaliteli şarkı embedding'leri, aynı modalitelere erişen doğrudan cosine baseline'dan daha iyi ve daha sağlam müzik benzerliği sıralaması sağlayabiliyor mu?*

## 2. Sabit başlangıç noktası

### 2.1. Mevcut dataset ve D4 deneyi

- Katalog: **13.822** MSD A–D şarkısı.
- Ses: başlangıç / orta / son bölümlerden 30 saniyelik pencereler, **26 kanallı** MSD segment özellikleri.
- Cache: `data/msd_hf_abcd_pilot/abcd_audio_start_mid_end_30s_26ch.h5`, yaklaşık **315,18 MiB**.
- Geçerli pencereler: başlangıç **13.822**, orta **13.813**, son **13.813**; **13.804** şarkıda 3, **18** şarkıda 2 geçerli pencere var.
- Değerlendirme: **3 artist-grouped fold × 3 seed = 9** D4 checkpoint'i. Seed'ler **684, 685, 686**.
- D4 eğitimi validation-only geliştirme protokolüyle tamamlandı; bağımsız test sonuçları üretilmedi.
- D4 üretim koşusu yaklaşık **15 saat 58 dakika** sürdü. Bu nedenle P0 kapsamında yeniden eğitimden kaçınılacak.
- D4'ün final test raporu **546 geçen birim test** kaydeder; P1-A son preflight'ında **578 test** geçti, `compileall`, `pip check` ve CLI help kontrolleri başarılı oldu.

### 2.2. Rapordan aktarılan validation sonuçları

| Yöntem | Gözlem | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| D0 neural | 9 fold-seed | 0.119240 | 0.194019 | 0.072913 | 0.071530 |
| D4 neural | 9 fold-seed | **0.127762** | **0.205434** | **0.076902** | **0.075412** |
| D0-matched cosine | 3 fold | 0.121607 | 0.204624 | 0.087984 | 0.077921 |
| D4-matched cosine | 3 fold | **0.128981** | **0.219120** | **0.089028** | **0.080621** |

**Okuma:** D4 neural, D0 neural'den yüksek ortalama alıyor; 9 eşleştirilmiş seed karşılaştırmasının 8'inde D4 lehine NDCG artışı var. Ancak **D4-matched cosine dört metrikte de D4 neural'in önünde**. Bu sonuçların tümü validation-only'dir; checkpoint seçimi için aynı validation kullanılmıştır. Dokuz koşuyu dokuz bağımsız test gibi yorumlama.

### 2.3. Eski D modelinden gelen duyarlılık referansı

Önceki **tek pencereli D** modelinde, giriş sıfırlama sonucu NDCG@10 düşüşleri:

| Sıfırlanan girdi | NDCG@10 düşüşü | Maskesizden farklı kalan Top-10 örtüşmesi |
|---|---:|---:|
| Genre | 0.048561 | 0.265619 |
| Lyrics | 0.018911 | 0.124028 |
| Audio | 0.007963 | 0.586110 |
| BPM | 0.001268 | 0.861922 |

Bu değerler **D4'e ait değildir**. D4 için aynı prosedürle yeni ölçüm yapıldıktan sonra karşılaştırılabilir. Sıfırlama dağılım dışı bir müdahaledir; gerçek nedensel önem puanı değildir.

---

## 3. P0-A: D4 modalite duyarlılığı (yeniden eğitim yok)

### Amaç

D4'ün çok pencereli audio kapsamı, audio kolunun **sıralamaya etkisini** D0/D'ye kıyasla artırdı mı? Genre/lyrics baskınlığı devam ediyor mu?

### Sabit deney tasarımı

- **9 mevcut D4 checkpoint'ini salt okunur yükle:** fold `0/1/2`, seed `684/685/686`.
- Her checkpoint için **5 koşul** çalıştır: `unmasked`, `audio_zero`, `bpm_zero`, `genre_zero`, `lyrics_zero`.
- Tüm koşullar **aynı validation query/gallery ve aynı bilinen pozitif bağlantı kümesini** kullansın.
- Audio maskesinde pencere geçerliliği, segment uzunlukları ve padding maskesi değişmesin; yalnız normalize edilmiş özellik değerleri sıfırlansın.
- Diğer sıfırlama işlemlerinde modalite boyutları, normalizasyon parametreleri ve eksik veri işleme değişmesin.
- Fold normalizasyon/IDF/genre sözlükleri yalnızca ilgili train bölümünden gelsin.
- **Test track'lerine ait embedding, relevance ve test metriği hesaplama.**

### Çıkış ölçümleri

Her `fold × seed × koşul` kaydı için:

- `Recall@10`, `HitRate@10`, `MRR@10`, `NDCG@10`
- `delta_ndcg = ndcg_unmasked - ndcg_masked`
- Maskesiz / maskeli **Top-10 küme örtüşmesi** (aynı sorgular için)
- Ortalama **embedding cosine kayması** = `mean(1 - cosine(z_original, z_masked))`
- Query sayısı, pozitif bağlantı sayısı, model/checkpoint/hash bilgisi

**Önerilen artefaktlar (yeni; mevcut dosya oldukları varsayılmamalı):**

```text
results/neurotune_v1_d4_modality_sensitivity/
  experiment.json
  condition_records.csv
  paired_d0_d4_comparison.csv
  README.md
```

### D0–D4 eşleştirilmiş analiz

- D0 ve D4 modalite analizinde **aynı fold, seed, query/gallery, etiket kaynağı ve maskeleme semantiği** kullanıldığını doğrula; farklıysa doğrudan paired fark üretme.
- Her modalite için `delta_ndcg_D4 - delta_ndcg_D0` hesapla; mutlak ve göreli düşüşü ayrı göster.
- Fold düzeyinde ortalamayı ve seed dağılımını raporla. Gerekirse fold başına üç seed'in ortalamasını al, **n=9 bağımsız test** gibi p-değeri üretme.
- D4 audio düşüşü artarsa bunu yalnızca "mevcut maskeye daha duyarlı" diye yorumla; **daha iyi nedensel audio öğrenimi** diye sunma.

### Dağılım dışı müdahale kontrolü (opsiyonel, ilk koşudan sonra)

Sıfır maskeleme bulgusu kritikse, validation'da train dağılımına daha yakın **aynı türden/aynı katalogdaki başka bir şarkının modalite vektörüyle değiştirme** deneyi ekle. Bu bir *permutation/swap sensitivity* analizidir; yine nedensellik ispatı değildir. Değiştirilen vektörü seed ile deterministik seç; diğer modaliteleri sabit tut. Bu kontrolü ilk P0 teslimini bloke eden zorunlu adım yapma.

### Kabul kriterleri

- 9 × 5 = **45 benzersiz koşul kaydı** ve tüm metriklerin sonlu olması.
- 9 maskesiz D4 validation sonucu, kayıtlı D4 `experiment.json` sonucuyla tolerans içinde eşleşmeli.
- Orijinal D4 checkpoint'leri, cache ve D0–D3 artefaktları değişmemeli; çalışmadan önce/sonra hash doğrulanmalı.
- Yeni runner otomatik testlerinde maske semantiği, boyutlar, fold izolasyonu ve parity kontrolü yer almalı.

---

## 4. P0-B: Triplet ve embedding diagnostics (yeniden eğitim yok)

### Amaç

Neural modelin cosine baseline'ı geçememesinin olası nedenlerini ayırt etmek:

1. Triplet'ler çok mu kolay?
2. Öğrenilmiş embedding'ler ayrım yapabiliyor mu?
3. Model audio yerine metadata'ya mı bağımlı?
4. Eğitim ile validation mesafeleri arasında belirgin fark var mı?

### Veri sınırları ve gerekli ayrım

- **Train diagnostics:** Yalnızca train triplet schedule'ları ve train örnekleri; train etiketlerini kullanabilir.
- **Validation diagnostics:** Validation üzerinde mevcut etiketler ve mevcut checkpoint'ler; yalnızca tanısal ve model geliştirme amaçlıdır.
- **Test:** Kullanılmayacak. Resmî/legacy test pozitiflerinden eğitim hedefi üretme; mevcut eski negative-exclusion politikasını değiştirdiğini iddia etme.
- Bilinmeyen bağlantıları "gerçek negatif" diye etiketleme. *Sampled negative* terimini kullan.

### Kaydedilecek tanılar

| Ölçüm | Açıklama | Beklenen kullanım |
|---|---|---|
| `d(a,p)` dağılımı | Anchor–positive cosine distance | Pozitiflerin ne kadar yakınlaştığını görmek |
| `d(a,n)` dağılımı | Anchor–sampled-negative cosine distance | Negatif ayrımı |
| `triplet_gap = d(a,n)-d(a,p)` | Örnek bazlı ayırma marjı | Margin yakınındaki örnekler |
| Active-triplet oranı | `loss > 0` oranı; margin=0.2 varsayımı rapordan doğrulanmalı | Kolay triplet baskınlığı var mı? |
| Easy / semi-hard / hard payları | `gap >= m` / `0 < gap < m` / `gap <= 0` | Gelecek mining kararları |
| Embedding pairwise cosine | Şarkıların embedding dağılımı | Aşırı yakınlaşma/çökme belirtisi |
| Per-modality norm/etki | Encoder/fusion aktivasyonları, mümkünse | Kol baskınlığı hipotezi |
| Fold/seed dispersion | D4 ve D0 metrik dağılımı | Eğitim kararlılığı |

**Önemli not:** D4'ün D0 schedule kopyalarını kullandığı raporlanıyor. Bu, eşleştirilmiş teşhis için avantajdır; ancak schedule kayıtlarının gerçekten aynı örnek sırası ve kapsamı taşıdığını hash / örnek ID üzerinden doğrula.

### Aktif triplet hesabı

Triplet margin loss kullanılıyorsa:

```text
d_ap = 1 - cosine(anchor_embedding, positive_embedding)
d_an = 1 - cosine(anchor_embedding, sampled_negative_embedding)
loss = max(0, d_ap - d_an + margin)
active = (loss > 0)
```

Mevcut eğitimin efektif margin ve distance implementasyonunu koddan doğrula; yukarıdaki formülü kodun yerine körlemesine geçirme.

### Önerilen çıktı

```text
results/neurotune_v1_d4_triplet_diagnostics/
  experiment.json
  triplet_summary_by_fold_seed.csv
  distance_distributions.csv
  analysis.md
```

Tüm eğitim triplet'lerini yeniden üretmek/forward etmek pahalıysa, **sabit seed ile örneklenen, manifest'e yazılan temsilî alt küme** kullan. Alt küme büyüklüğünü, örnekleme yöntemini ve kaç kayıt atlandığını rapora ekle. P0'un amacı 16 saatlik eğitimi tekrarlamak değil.

### Kabul kriterleri

- Train ve validation teşhisleri ayrı tablolarda.
- Her fold'da seed, örnek sayısı, margin, embedding normalizasyonu ve kullanılan checkpoint belirtilmiş.
- Eksik/çözümlenemeyen sample ID veya non-finite distance sessizce geçilmiyor.
- D4 ve D0 karşılaştırmasında aynı örnekler, aynı distance ve aynı negatif kapsamı kullanılıyor veya fark açıkça belgeleniyor.
- Sonuçlardan **yeni bir modelin daha iyi olduğu iddia edilmiyor**.

---

## 5. P1-A: Genre içi ve metadata kontrollü retrieval

### Amaç

Genre sıralamayı güçlü etkiliyor; peki model **aynı tür içindeki müzikal benzerliği** ayırt edebiliyor mu?

### Deney tasarımı

1. D0 neural, D0-matched cosine, D4 neural ve D4-matched cosine'ı **aynı validation candidate policy** ile değerlendir.
2. Her sorgu için ham katalog genre metninin tam eşitliğine göre gallery kur; sorgunun kendisini çıkar. Train-fitted one-hot genre özelliğini filtreleme için kullanma.
3. Pozitifleri yalnız `validation_edges` train-source bağlantılarından ve yalnız aynı-genre gallery içinde al. Test kaynaklı kenarlar yalnız loader'ın bütünlük doğrulamasında kalır.
4. Pozitif bağlantının aynı-genre gallery içinde bulunabildiği **eligible sorguları** ve uygun olmayan sorgu nedenlerini ayrı raporla.
5. Gallery boyutu 10'dan küçük eligible sorguları ana analizde tut; alt grubu ayrıca raporla.
6. Artist dışlama ek koşulu bu tamamlanmış temel analizde kullanılmadı.

### Raporlanacaklar

- Eligible query sayısı, ilgili pozitif kenar sayısı, gallery büyüklüğü dağılımı.
- `Recall@10`, `MRR@10`, `NDCG@10`; macro-query, eşit ağırlıklı genre-macro ve micro Recall@10.
- Genre başına skorlar; <10 gallery alt grubu ve missing genre / no candidates / no in-gallery positive sayıları.
- Genre-only/metadata-only referans bu koşuda hesaplanmadı; aynı gallery ve etiketlerle ayrıca kurulmadan karşılaştırmaya eklenmemeli.

**Uyarı:** Genre içi filtreleme problemi değiştirir; sonuçları ana tam-katalog retrieval metriğiyle doğrudan yan yana "iyileşme" diye yorumlama.

### Sonuçlar (23 Kasım 2026)

Çalışma yeniden eğitim yapmadan **135,107 saniyede** tamamlandı. 3 fold'da toplam **4.093 fold-query gözlemini** kapsadı: 2.081 eligible sorgu, 2.010 sorguda aynı-genre gallery içinde bilinen pozitif bulunmadı, 2 sorguda candidate yoktu ve kullanılabilir genre'si olmayan sorgu yoktu. Eligible gallery boyutu medyanı 672, ortalaması 428,92 idi; yalnızca 4 eligible sorgunun gallery'si 10'dan küçüktü. Fold'lar boyunca sayılan bilinen in-genre pozitifler 2.306 benzersiz undirected kenar (4.612 directed query-relevance çifti) oluşturdu. Üretilen query ID'leri üç fold'un validation ID kümeleriyle tam eşleşti; held-out test ID'leriyle kesişim yoktu.

| Sistem | Gözlem | Macro-query R@10 / MRR@10 / NDCG@10 | Genre-macro R@10 / MRR@10 / NDCG@10 | Micro R@10 |
|---|---:|---|---|---:|
| D0 neural | 9 fold-seed | 0,1316 / 0,0781 / 0,0782 | 0,3483 / 0,1959 / 0,2157 | 0,1096 |
| D4 neural | 9 fold-seed | 0,1407 / 0,0819 / 0,0820 | 0,3564 / 0,1842 / 0,2084 | 0,1204 |
| D0-matched cosine | 3 fold | 0,1371 / 0,0942 / 0,0861 | 0,3540 / 0,2137 / 0,2230 | 0,1158 |
| D4-matched cosine | 3 fold | 0,1472 / 0,0954 / 0,0896 | 0,3669 / 0,2071 / 0,2211 | 0,1256 |

**Yorum:** D4 neural'in macro-query NDCG@10 ortalaması D0'dan **0,003854** yüksekti; 9 eşleşmiş fold-seed karşılaştırmasının 8'inde pozitifti ve üç fold'un ortalama farkı da pozitifti. Buna karşılık eşit ağırlıklı genre-macro NDCG@10 D4'te **0,007272 daha düşüktü**. D4 neural'in macro-query NDCG@10 ortalaması D4-matched cosine'dan **0,007569 daha düşüktü**; D4 neural yalnızca fold 0'da bu cosine referansını geçti. Sonuç bu nedenle aggregation ve genre'ye göre değişiyor; model üstünlüğü veya genelleme kanıtı değildir. Aynı-genre pozitif Last.fm bağlantıları eksik olabilir ve validation kullanılmıştır; bağımsız test değerlendirmesi yapılmadı. Ayrıntılı per-query/per-genre sonuç ve provenance: `results/neurotune_v1_p1a_intra_genre_retrieval/`.

---

## 6. P1-B: Audio branch teşhisi ve isteğe bağlı audio-only kontrol

### Amaç

D0 ile D4'ün fusion öncesi audio-branch temsillerini aynı validation galerilerinde karşılaştırmak; çok pencereli D4 kolunun retrieval sinyali taşıyıp taşımadığına dair tanı elde etmek. Bu, multimodal checkpoint'lerden çıkarılan branch ölçümüdür; bağımsız audio-only eğitim veya nedensel katkı testi değildir.

### İki aşamalı strateji

**Önce eğitim gerektirmeyen tanı:**

- D0 ve D4 frozen multimodal checkpoint'lerinin fusion öncesi 64-D `audio_projection` çıkışlarını al.
- Aynı validation gallery için L2 normalize ederek cosine retrieval hesapla.
- Sonuçları **D0/D4 audio-branch diagnostic** olarak etiketle; bu temsilleri bağımsız eğitilmiş audio-only modellerle eşdeğer sayma.
- P1-A'nın ham genre metniyle oluşturduğu aynı-genre validation galerilerini, aynı query/pozitif politika ve hash'leriyle kullan.
- D0 ve D4'ü tam `(fold, seed)` kimliğiyle eşleştir; held-out ABCD test ID'lerini feature building, query/gallery, relevance veya metric hesaplamasına sokma.
- MXM kaynakları provenance için hash'lenir; özellik kurucusuna yalnız ilgili fold'un train+validation catalog ID'leri verilir. D0/D4 checkpoint/schedule'ları önceki official/legacy pozitiflerin negatif aday elemesinde yalnızca exclusion amaçlı kullanımını miras alır; P1-B yeni negatif örneklemez ve eğitim yapmaz.

**Yalnızca gerekirse yeni kontrol eğitimi:**

- Yeni bir **multi-window audio-only** model tanımla ve ayrı deney namespace'inde eğit.
- Tek değişken pencere kapsamı olacak şekilde aynı train/validation split, seed, triplet policy, budget ve optimizer koşullarını koru.
- CPU süresini göz önünde bulundurarak önce düşük maliyetli pilot yapılabilir; **pilotu seçici biçimde kullanmanın validation yanlılığı oluşturabileceğini** raporla.
- Önceki audio-only koşuların farklı protokollerinin metriklerini yeni koşuyla doğrudan kıyaslama.

### Başarı tanımı

Audio branch'lerin retrieval davranışına dair eşleştirilmiş, validation-only bir *teşhis* elde etmek. Bu ölçüm model üstünlüğü veya genelleme kanıtı değildir; tek başına yeni uzun eğitimi ya da ürün dağıtımını onaylamaz.

### Sonuçlar (24 Kasım 2026)

Koşu frozen D0/D4 checkpoint'leriyle **123,79 saniyede**, **3.475.596 KiB** tepe RSS kullanarak tamamlandı; eğitim yapılmadı. Dokuz fold-seed eşleşmesinde 18 branch koşulu ve 12.486 eligible per-query sistem satırı üretildi. D0/D4 branch'lerinin validation ID, candidate ve label hash'leri P1-A ile eşleşti; 11 önceki sonuç namespace'i değişmeden kaldı.

| Sistem | Gözlem | Recall@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| D0 audio branch diagnostic | 9 fold-seed | 0,106647 | 0,057007 | 0,058571 |
| D4 audio branch diagnostic | 9 fold-seed | 0,120503 | 0,067596 | 0,067903 |
| D0 neural (P1-A referansı) | 9 fold-seed | 0,131641 | 0,078092 | 0,078158 |
| D4 neural (P1-A referansı) | 9 fold-seed | 0,140735 | 0,081916 | 0,082013 |
| D4-matched cosine (P1-A referansı) | 3 fold | 0,147165 | 0,095410 | 0,089582 |

**Yorum:** D4 audio-branch macro-query NDCG@10, D0 branch'inden ortalama **+0,009331** yüksekti; 9 eşleşmenin 8'inde artış vardı. Fold ortalama farkları +0,012888 / +0,001296 / +0,013809 oldu; fold 1 / seed 685 eşleşmesi −0,014844 ile ters yöndeydi. D4 branch, D4 neural referansından **0,014110**, D4-matched cosine'dan **0,021679** daha düşük NDCG@10 aldı. Bu, çok-pencere branch'inin D0 branch'ine göre daha iyi tanısal sıralama verdiğini gösterir; tek başına pencerelerin nedensel katkısını, model üstünlüğünü veya genellemeyi kanıtlamaz. Ders ödevi kapsamında sonuç yeterlidir; ek uzun audio-only eğitim veya P2 kaynak araştırması başlatılmayacak.

Çıktılar: `results/neurotune_v1_p1b_audio_branch_diagnostic/` (`experiment.json`, `metrics.csv`, `per_query_metrics.csv`, `fold_query_coverage.csv`, `analysis.md`). Metadata düzeltmesi yalnız provenance etiketi/sınır açıklamasını değiştirdi; run-time code hash'i ve sonuç tabloları korundu. Final preflight: **588 test geçti**, `compileall`, `pip check`, P1-B CLI/py_compile kontrolleri geçti; bağımsız kod incelemesinde bulgu çıkmadı.

---

## 7. P2: Bağımsız test ve metodoloji temizliği

### Neden?

D4 raporundaki metrikler validation-only. Ayrıca önceki D0 politikasından taşınan **official/legacy test-source pozitifleri negatif aday elemesinde kullanılabilmiş**. Bu durum, mevcut deney için belgelenmiş bir sınırlılıktır; yeni bir test üzerinde tarafsız genelleme iddiası kurmak için daha katı izolasyon gerekir.

### Yeni test için gereksinimler

- **Artist-disjoint** kohort: geliştirme sürecinde kullanılan train/validation artist'larından bağımsız.
- Mümkünse **yeni veya bağımsız similarity etiket kaynağı**; değerlendirme birimi ve Last.fm/insan tercihi farkı açık.
- Yeni test şarkılarının **feature normalization, IDF, genre vocabulary, checkpoint seçimi, negative sampling/mining ve hiperparametre seçiminde** hiçbir etkisi olmayacak.
- **Test pozitiflerini negatif seçiminden filtreleme amacıyla bile kullanma.** Negatif belirsizliğini açıkça raporla; gerekirse pozitif ve unlabeled yaklaşımı seç.
- Test query/gallery/positive-edge sayıları ve label density önceden raporlanmalı. Çok seyrek testte tek bir Recall@10 ortalamasına aşırı anlam yükleme.
- Veri kaynağı provenansı ve lisans doğrulaması belgelensin. Bu, önceki A–D arşivleri için de ayrı yayın/dağıtım kontrolüdür.
- Bütün deney tasarımı ve kararlar **test açılmadan önce** dondurulsun.

### Önceden seçilen kabul kuralı (24 Kasım 2026)

- Birincil karşılaştırma frozen D4 neural ile aynı query/gallery üzerindeki D4-matched cosine arasındadır.
- P2 primary gallery, bağımsız test cohort'undaki tam katalogdur: her eligible parça query, diğer tüm eligible test parçaları candidate; query'nin kendisi çıkarılır, genre filtresi uygulanmaz.
- Makro-query NDCG@10 farkı en az **+0,01 mutlak** olmalı ve eşleştirilmiş artist-cluster bootstrap %95 güven aralığının alt sınırı **0'dan büyük** olmalıdır.
- D4 neural ve cosine aynı test query'lerini, candidate gallery'lerini ve relevance etiketlerini kullanır. Recall@10, MRR@10 ve HitRate@10 ikincil/descriptive metriklerdir.
- Tam uygulama taslağı `docs/plans/2026-11-24-neurotune-p2-independent-test-protocol.md` içindedir. İlk kaynak/lisans taraması `docs/plans/2026-11-24-neurotune-p2-test-source-research.md` dosyasına kaydedildi; henüz uygun bir dataset seçilmedi ve test etiketlerine erişilmedi.

### İstatistik ve raporlama

- Model–baseline farklarını **aynı fold/query** düzeyinde eşleştirerek raporla.
- Seed'leri bağımsız test seti kabul etme. İmkân varsa query veya artist düzeyindeki bootstrap için bağımlılık yapısını dikkate al.
- Önceden belirlenmiş ana metrik **NDCG@10**; ikinciller Recall@10, MRR@10, HitRate@10.
- Yeni test bulunamıyorsa final raporunda yalnızca validation bulgularını ve kısıtları sun; genelleme/üstünlük iddiası kurma.
- İlk resmî kaynak/lisans taramasında doğrudan seçilebilir temiz bir dataset bulunmadı; candidate matrix ve sınırlılıklar `docs/plans/2026-11-24-neurotune-p2-test-source-research.md` içindedir. Kullanıcı bu projenin ders ödevi olduğunu ve daha fazla kaynak/lisans araştırmasına gerek olmadığını belirtti; P2 kaynak arama ve bağımsız test bu çalışma kapsamında ertelendi.
- İlk resmî kaynak/lisans taramasında doğrudan seçilebilir temiz bir dataset bulunmadı; candidate matrix ve sınırlılıklar `docs/plans/2026-11-24-neurotune-p2-test-source-research.md` içindedir. Test verisi/etiketleri açılmadı.

---

## 8. Uygulama sırası ve görev listesi

> Aşağıdaki yol ve script adları öneridir; mevcut projede zaten bulundukları varsayılmamalıdır. Mevcut dosya düzenine göre adapte edilebilir.

### Sprint 1: P0-A (tamamlandı; read-only, yeni eğitim yok)

- [x] D4 `experiment.json`, dokuz checkpoint ve transform manifest/hash kontrolünü yap.
- [x] D modelindeki modalite duyarlılığı runner'ının D4 multi-window batch formatına uyarlanabilirliğini incele.
- [x] Beş koşulu ve 45 kayıt parity testlerini uygula.
- [x] Her koşul için retrieval + Top-10 overlap + embedding cosine shift hesapla.
- [x] D0–D4 paired modalite farklarını şartları doğrulayarak raporla.
- [x] Tam test paketini çalıştır ve yeni namespace'e sonuç yaz.

### Sprint 2: P0-B (tamamlandı; read-only, yeni eğitim yok)

- [x] Train schedule / feature / checkpoint uyumunu kontrol et.
- [x] Anchor-positive / anchor-negative distance dağılımlarını hesapla.
- [x] Active/easy/semi-hard/hard triplet oranlarını çıkar.
- [x] D0–D4 arasında eşleştirilmiş distance ve active-triplet karşılaştırması yap.
- [x] Eğitim zamanı ve bellek maliyetini not et; gerekiyorsa deterministik örnekleme kullan.
- [x] Sonuçları hem CSV hem kısa yorum raporu olarak yaz.

### Sprint 3: P1-A (yeni eğitim yok)

- [x] Same-genre eligibility ve gallery oluşturma kurallarını sabitle; ham genre metninde tam eşitlik kullan.
- [x] D0/D4 neural ve eşleşmiş cosine sistemlerinin genre içi retrieval metriklerini hesapla.
- [x] Query ve pozitif kenar kapsamını, <10 gallery alt grubunu ve selection-bias sınırlılığını raporla.

### Sprint 4: P1-B (önce read-only)

- [x] D0/D4 audio branch embedding'lerinden aynı P1-A validation galerilerinde cosine retrieval çıkar; exact fold-seed, ID/hash eşleşmesini ve held-out test sınırını doğrula.
- [x] Frozen checkpoint provenance, no-training sınırı, compact query metrics, eligibility coverage ve paired D4−D0 yorumunu yeni namespace'te kaydet.
- [ ] Gerekirse ayrı multi-window audio-only kontrollü eğitim için tasarım belgesi hazırla.
- [ ] Ancak yeni soru ve veri gerekçesi varsa kontrollü audio-only eğitim tasarla; mevcut P1-B bulgularıyla uzun eğitim başlatma.

### Sprint 5: P2 (bağımsız değerlendirme)

- [ ] Yeni artist-disjoint ve etiket bağımsızlığı olan test kaynağını belirle.
- [ ] Test leakage ve lisans/provenans kontrolünü kayda bağla.
- [ ] Model, checkpoint ve metrikleri test çalışmadan önce sabitle.
- [ ] Tek seferlik nihai değerlendirmeyi koş; sonuçları sınırlılıklarıyla birlikte raporla.

---

## 9. Kalite kapıları ve sonuçların sunumu

### Her sprint için zorunlu

- İzole **yeni output namespace**; var olan D0–D4 raporlarını/artefaktlarını overwrite etme.
- Değişmez kaynakları işlem öncesi ve sonrası hash ile doğrula.
- Train-only preprocessing ve artist-grouped fold sınırlarını koru.
- Reproducibility: seed, parametreler, kullanılan sorgu/gallery, kaynak etiket, commit veya code hash ve ortam bilgileri.
- `unittest`, `compileall`, `pip check` kontrolleri ve ayrıntılı hatalar.
- Ana metriklerde model/baseline **matched** karşılaştırması.
- Test verisini kullanan deneysel karar, yeni bağımsız test iddiasını geçersiz kılabilir; bu sınırı ihlal etme.

### Son raporda yer alacak ana tablolar

1. D0/D4 modalite duyarlılığı: NDCG düşüşü, overlap, cosine kayması.
2. D0/D4 triplet: easy/semi-hard/hard, active ratio, d(a,p)/d(a,n).
3. Full-gallery vs same-genre retrieval: query/edge coverage ve skorlar.
4. D4 audio-only branch teşhisi vs full D4 ve ilgili cosine referanslar.
5. Ayrı bağımsız test yapılabilirse: önceden dondurulmuş metriklerle final model-baseline karşılaştırması.

### Karar ağacı

| Bulgu | Sonraki karar |
|---|---|
| D4'te audio masking duyarlılığı artıyor, ancak genre baskınlığı sürüyor | Modality balancing/dropout'un D4'e taşınması **yeni ve ayrı** geliştirme deneyi olarak düşünülebilir |
| Triplet'lerin büyük çoğunluğu easy; pozitif/negatif ayrımı yetersiz | Negatif madenciliğini ve sampling gürültüsünü gözden geçir; eski D1 raporunu önce oku |
| Genre içi performans belirgin düşüyor | Audio encoder/label policy için daha hedefli tanı; genre dışı başarıyı tek başarı ölçüsü kabul etme |
| D4 audio branch D0'a göre gelişiyor | Multi-window'un audio temsilini güçlendirdiği hipotezi desteklenir; nedensellik iddiası kurulmaz |
| Neural model yine baseline'ın gerisinde | Daha büyük ağ yerine supervision/label kalitesi ve doğrudan cosine'ın güçlü olması tartışılmalı |
| Bağımsız test yapılamıyor | Ders raporunda validation-only sınırlılığı açık tutulmalı |

---

## 10. Agent için kısa uygulama talimatı

```text
NeuroTune deposunda 23.11.2026 D4 sonuçlarını temel al.

Önce yalnızca P0-A ve P0-B'yi uygula. Mevcut D4 checkpoint'lerini,
D0-D4 deney çıktılarını ve HDF5 cache'i değiştirme. Yeniden eğitim başlatma.

P0-A: 3 fold × 3 seed × 5 (unmasked + 4 tek-modalite maskesi)
validation koşusu; Recall@10, HitRate@10, MRR@10, NDCG@10,
Top-10 overlap ve embedding cosine shift; orijinal unmasked
sonuçlarla parity ve uygun koşullarda D0 paired comparison.

P0-B: Salt-okunur train/validation triplet diagnostics;
d(a,p), d(a,n), margin gap, active/easy/semi-hard/hard
oranları ve fold/seed dağılımları. Bilinmeyen eşleşmeleri kesin
negatif sayma. Gerekirse örneklemeyi deterministik yap.

Tüm çıktıları yeni, overwrite edilmeyen namespace'lere yaz.
Test-source verilerinden feature/embedding/relevance/metrik üretme.
Train-only normalizasyon ve fold izolasyonunu koru.

Önce tasarımı ve veri erişim sınırlarını kontrol et; sonra kodla,
test et ve doğrula. Yeni performans sayısı üretmediysen uydurma.
Sonunda hangi hipotezlerin desteklendiğini ve P1'in gerekip
gerekmediğini kısa, kanıta dayalı bir raporda belirt.
```

---

## 11. Kaynak notu ve sınırlar

Bu belge yeni deneylerin sonuçlarını değil, **23 Kasım itibarıyla doğrulanmış raporlardan türetilmiş sonraki iş planını** kaydeder. D4 sayıları `NeuroTune_D4_Son_Testler_2026-11-23.md` dosyasından; eski D modalite sıfırlama değerleri `NeuroTune_D_Modalite_Duyarliligi_2026-11-17.md` dosyasından aktarılmıştır. **P0–P2 deneyleri henüz bu belgede yapılmış sayılmaz.** D1–D3'ün ayrıntılı skorları bu temel D4 raporunda verilmediği için karşılaştırma sıralaması çıkarılmamıştır.
