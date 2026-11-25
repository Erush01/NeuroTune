# NeuroTune v1: Model Mimarisi, Literatür ve Deney Planı

**Ders:** CMP684 - Neural Networks  
**Proje:** Müzik benzerliği için çoklu özellikli şarkı embedding'leri  
**Durum:** Önerilen teknik tasarım (nihai veri kümesi ve sonuçlar henüz doğrulanmadı)  
**Tarih:** 12 Kasım 2026

## 1. Proje Amacı

Bir şarkının müzikal zaman serilerini temel alarak, ek olarak **BPM**, **lyrics** (şarkı sözleri) ve **genre** (müzik türü) bilgilerini kullanan bir **similar song retrieval** sistemi geliştirmek.

Her şarkı tek bir vektöre (song embedding) dönüştürülecek. Eğitimde benzerlik ilişkileri kullanılarak embedding uzayı öğrenilecek, sorgulama sırasında ise önceden hesaplanmış embedding'ler **cosine similarity** ile karşılaştırılarak **Top-K** şarkılar bulunacak.

### Araştırma sorusu

> Does combining temporal audio features with tempo, lyrical, and genre information improve music similarity retrieval using a compact neural embedding model?

**Ana karar:** Zaman serisi özellikleri modelin merkezi girdisidir. BPM, lyrics ve genre tamamlayıcı girdilerdir. Yalnızca basit istatistiklere indirgenmiş bir audio temsilini ana model olarak kullanmayacağız.

## 2. İlgili Makaleler

Bu bölüm, önceki literatür değerlendirmesinde ele alınan çalışmaları ve NeuroTune açısından kullanışlı fikirlerini özetler. Bazı ayrıntıların (özellikle veri kümesi sürümleri ve tam mimari konfigürasyonlarının) son uygulama öncesinde makale tam metinlerinden tekrar kontrol edilmesi gerekir.

### 2.1. Disentangled Multidimensional Metric Learning for Music Similarity (Lee vd., ICASSP 2020)

- **Referans:** https://doi.org/10.1109/ICASSP40776.2020.9053442
- **Veri:** Million Song Dataset (MSD) çevresinde oluşturulmuş müzik verileri; Dim-Sim ile benzerlik değerlendirmesi.
- **Girdi:** Gerçek sesin kısa parçalarından hesaplanan log-mel spectrogram.
- **Model:** CNN/Inception tabanlı audio encoder ve çok boyutlu metric learning.
- **Öğrenme:** Conditional Triplet Loss.
- **Benzerlik yönleri:** Genre, mood, instrument, tempo gibi ayrıştırılmış boyutlar.
- **NeuroTune'ye katkısı:** Şarkıları önceden hesaplanabilir embedding'lere dönüştürmek ve benzerlik ilişkilerinden öğrenmek.

**Almayacağımız kısım:** Büyük Inception mimarisi ve gerçek ses dosyası gerektiren spectrogram işleme hattı.

### 2.2. Multi-Input Architecture and Disentangled Representation Learning for Multi-Dimensional Modeling of Music Similarity (Ribecky vd., AES 2022)

- **Referans:** https://arxiv.org/abs/2111.01710
- **Veri:** MSD tabanlı örnekler ve Dim-Sim türü insan benzerlik karşılaştırmaları.
- **Girdiler:** Mel-spectrogram, chromagram ve tempogram gibi farklı müzik temsilleri.
- **Model:** Birden fazla giriş kolu kullanan CNN encoder'lar.
- **Öğrenme:** Conditional Triplet Loss.
- **NeuroTune'ye katkısı:** Farklı müzikal bilgi türlerini ayrı encoder'larda işleyip sonradan birleştirmek.

**Almayacağımız kısım:** Ağır çok girişli CNN/Inception tasarımı. MSD'nin HDF5 segment özellikleri, bu makalelerdeki gerçek ses tabanlı spectrogram'larla aynı veri değildir.

### 2.3. Probabilistic Approaches for Music Similarity Using Restricted Boltzmann Machines (Tran, Ngo ve d'Avila Garcez)

- **Referans:** https://doi.org/10.1007/s00521-019-04106-y
- **Veri:** MagnaTagATune üzerine kurulu müzik benzerliği değerlendirmeleri.
- **Model:** Restricted Boltzmann Machine (RBM) tabanlı olasılıksal yaklaşım.
- **Hedef:** Şarkılar arasındaki göreli benzerliği modellemek.
- **NeuroTune'ye katkısı:** Benzerliği tek şarkı sınıflandırması yerine şarkı çiftleri/üçlüleri arasında öğrenilen bir ilişki olarak ele almak.

**Karar:** RBM'yi NeuroTune'nin ana encoder'ı olarak kullanmayacağız.

### 2.4. Music Similarity Measurement and Recommendation System Using Convolutional Neural Networks (Sheikh Fathollahi ve Razzazi, 2021)

- **Referans:** https://doi.org/10.1007/s13735-021-00206-5
- **Model:** CNN ile müzik temsili çıkarma ve özellik uzayında benzerlik hesabı.
- **Yaklaşım:** CNN'den elde edilen özelliklerle cosine/Euclidean tabanlı retrieval.
- **NeuroTune'ye katkısı:** Eğitim ve sorgulama maliyetlerini ayıran, embedding + vektör benzerliği mimarisi.
- **Not:** Tam veri kümesi listesi ve kullanılan katman yapısı, uygulama öncesinde makale tam metninden doğrulanmalıdır.

### 2.5. Literatürden çıkardığımız tasarım ilkeleri

| Çalışma | Temel fikir | NeuroTune'ye uyarlama |
|---|---|---|
| Lee vd. | Ayrıştırılmış müzik benzerliği, triplet metric learning | Embedding eğitimi ve triplet ilişkileri |
| Ribecky vd. | Farklı müzik temsilleri için ayrı encoder'lar | Audio, BPM, lyrics, genre kolları |
| Tran vd. | Göreli şarkı benzerliği | Pozitif/negatif ilişkiyle öğrenme |
| Fathollahi & Razzazi | Neural feature extraction ve nearest-neighbor retrieval | Önceden hesaplanan embedding + Top-K arama |

## 3. NeuroTune v1 Mimarisi

### 3.1. Genel veri akışı

```text
MSD segment-level audio time series
  |  pitches (T x 12)
  |  timbre  (T x 12)
  |  loudness (T x 1)
  |  segment duration (T x 1)
  v
Temporal 1D-CNN + masked/temporal pooling
  v
Audio embedding (64D)

BPM ----------> Tempo encoder (8D) -----------+
Lyrics BoW ---> Lyrics encoder (32D) ---------+--> Concatenate (120D)
Genre --------> Genre encoder (16D) ----------+          |
Audio embedding (64D) ------------------------+          v
                                               Fusion MLP
                                               120 -> 128 -> 64
                                                    |
                                               L2 normalization
                                                    |
                                               64D song embedding
                                                    |
                                               Cosine similarity
                                                    |
                                               Top-K similar songs
```

Boyutlar **başlangıç konfigürasyonu**dur; hiperparametre deneyiyle değiştirilebilir.

### 3.2. Temporal audio input

Başlangıçta kullanılacak kanallar:

| MSD alanı | Boyut | İşlev |
|---|---|---|
| `segments_pitches` | `T x 12` | Pitch/chroma aktivasyonları |
| `segments_timbre` | `T x 12` | Tını değişimi |
| `segments_loudness_max` | `T x 1` | Segment bazında dinamik seviye |
| `segments_start` | `T` | Segment başlangıç zamanları |
| `segments_start` farklarından türetilen süre | `T x 1` | Düzensiz aralıkları modelleme |
| `beats_start`, `sections_start` | Değişken | İleride ritim/yapı özelliklerine genişletme |

**Önerilen girdi matrisi:** `X_audio in R^(T x 26)`.

**Önemli:** Bu özellikler gerçek waveform veya mel-spectrogram değildir. MSD 10K HDF5 dosyalarındaki analiz özellikleridir.

MSD'de segment süreleri eşit olmadığından seçenekler:

1. Özellikleri düzenli zaman ızgarasına yeniden örneklemek (v1 için önerilen çözüm).
2. Segment sürelerini ayrı kanal olarak taşımak ve masked pooling kullanmak.

Şarkı uzunluklarının farklı olması nedeniyle sabit uzunluklu zaman pencereleri, padding/masking veya çoklu pencere birleştirmesi uygulanacak. İlk aşamada **30 saniyelik pencere** kullanılabilir; pencere boyutu deney parametresidir.

### 3.3. Temporal encoder

Başlangıç mimarisi:

```text
Input: T x 26
Conv1D(in=26, out=32, kernel=5) + ReLU
Conv1D(in=32, out=64, kernel=5) + ReLU
Conv1D(in=64, out=64, kernel=3) + ReLU
Masked temporal pooling
Projection -> Audio 64D
```

- Model küçük tutulacak; ilk sürümde Transformer kullanılmayacak.
- Global pooling, **CNN'den sonra** yapılacak; önce ham time-series ortalaması alınmayacak.
- Uzun dönem ilişkilerin etkisini anlamak için CNN+GRU yalnızca opsiyonel genişletme olacaktır.

### 3.4. Yardımcı encoder'lar

| Girdi | Ön işleme | Başlangıç encoder'ı | Çıkış |
|---|---|---|---|
| BPM | Train-only normalizasyon, geçersiz değer filtresi | Küçük MLP | 8D |
| Lyrics | musiXmatch BoW; train-only TF-IDF / sparse giriş | Linear + ReLU veya küçük MLP | 32D |
| Genre | tagtraum CD2C; one-hot/multi-hot | Linear + ReLU | 16D |

**Lyrics sınırı:** musiXmatch MSD araştırma verisi çoğunlukla tam söz/cümle sırası yerine kelime frekansı sağlar; bu yüzden v1 sözlerin derin anlamsal yapısını modellemez.

### 3.5. Fusion ve son temsil

```text
Audio 64D + BPM 8D + Lyrics 32D + Genre 16D = 120D
120D -> Fully Connected(128) -> ReLU -> Fully Connected(64)
L2 Normalize -> 64D Song Embedding
```

Ağ **tek bir şarkının** özelliklerini alıp tek bir embedding üretir. Siamese/Triplet eğitimde aynı ağ, birden çok şarkı için **paylaşılan ağırlıklarla** kullanılır; ayrı eğitilen ağlar değildir.

## 4. Öğrenme Hedefi

### 4.1. Triplet Metric Learning

Bir triplet:

- **Anchor (a):** Benzerleri aranacak şarkı.
- **Positive (p):** Benzerliği destekleyen dış ilişkiye göre benzer şarkı.
- **Negative (n):** Daha az benzer olduğu varsayılan, dikkatle örneklenmiş şarkı.

Temel loss:

```text
L = max(0, d(a, p) - d(a, n) + margin)
```

Burada `d` cosine distance, `margin` ayarlanabilir marj değeridir.

### 4.2. Etiket kaynağı ve dikkat edilmesi gerekenler

- **Last.fm similar-track ilişkileri** pozitif adayları oluşturabilir.
- İlişkinin bulunmaması kesin negatif kanıtı değildir; negatifler kontrollü seçilecek.
- Aynı şarkının farklı kaynaklardaki kimlikleri ve olası duplicate kayıtları temizlenecek.
- Aynı sanatçıya veya benzer tür etiketlerine yönelik sampling bias analiz edilecek.
- Kullanılan Last.fm ilişkilerinin, girişteki Last.fm türevi genre/tag bilgisiyle döngüsel değerlendirme üretip üretmediği kontrol edilecek.

**Başlangıç önerisi:** Offline seçilmiş pozitif ilişkiler + rastgele ve gerektiğinde yarı-zor negatif örnekler; tüm olası şarkı çiftlerini oluşturmayacağız.

## 5. Veri Kümesi Planı

### 5.1. Kaynaklar

| Kaynak | Kullanım | Bağlantı |
|---|---|---|
| MSD 10K Subset | Segment düzeyinde pitches, timbre, loudness, beat/section bilgisi ve tempo | https://millionsongdataset.com/ |
| musiXmatch MSD lyrics | Sözcük frekansı/BoW | https://millionsongdataset.com/musixmatch/ |
| tagtraum MSD Genre | Genre etiketleri | https://www.tagtraum.com/msd_genre_datasets.html |
| MSD Last.fm Dataset | Similar-track ilişkileri | https://millionsongdataset.com/lastfm/ |

**Önemli veri kısıtı:** MSD summary tek başına segment-level time-series içermez; bunlar için şarkı bazlı HDF5 analiz kayıtları gereklidir. Önce **MSD 10K** ile başlanacak.

### 5.2. Başlangıç veri kontrolü (M0)

Önce kaynakları `track_id` üzerinden eşleştirip şu sayıları raporla:

1. Segment time-series verisi geçerli olan şarkı sayısı.
2. Bunlar arasında BPM, lyrics ve genre verisi de bulunan şarkı sayısı.
3. Ortak kümede kalan pozitif Last.fm ilişki sayısı.
4. Pozitif ilişkisi bulunan benzersiz sorgu şarkısı sayısı.
5. Train / validation / test içinde değerlendirilebilir sorgu sayısı.
6. Sanatçı ve genre dağılımı; eksik veriler ve duplicate oranları.

**Henüz doğrulanmadı:** Dört kaynağın MSD 10K üzerindeki kesişimi. 10.000 şarkının tamamının eğitim için kullanılabileceği varsayılmamalı.

### 5.3. Fallback yaklaşımı

10K subset'te pozitif ilişki sayısı veya çoklu özellik kesişimi yetersizse:

- Daha büyük MSD metadata + lyrics + genre + Last.fm kesişimini bul.
- Bu kesişimdeki şarkılar için erişilebilir ayrıntılı HDF5 analiz verilerini edin.
- Çok büyük veri indirmeden önce dosya erişimi ve örnek sayısını pilotla doğrula.

Bu iş akışı, tam MSD arşivine hemen ihtiyaç duyulmadan veri kapsamını büyütmek içindir; istenen tüm HDF5 dosyalarının tek tek kolay erişilebilir olduğu varsayılmamalıdır.

## 6. Eğitim, Validation ve Test Protokolü

1. Şarkı kimlikleri ve mümkünse sanatçı grupları üzerinden ayrım yapılacak.
2. Pencere/segment düzeyindeki örnekler, aynı şarkı hem train hem testte olacak şekilde bölünmeyecek.
3. Normalizasyon, TF-IDF ve diğer veriyle öğrenilen ön işlemler **yalnızca training** verisine fit edilecek.
4. Model seçimi validation üzerinde yapılacak; test seti en son bir kez kullanılacak.
5. Retrieval aday havuzu ve değerlendirme sorguları açıkça tanımlanacak.
6. Sorguların test kümesinde ground-truth pozitif karşılıkları bulunduğu kontrol edilecek.
7. Negatif örnekleme yalnızca train ilişkilerinden ve train şarkılarından üretilecek; test ilişkileri eğitime sızmayacak.
8. Tekrarlı seed'ler, sabit konfigürasyon ve kayıtlı split manifest dosyaları kullanılacak.

**Değerlendirme sınırı:** Last.fm ilişkileri insan tercihinin eksiksiz ölçümü değildir. Ölçülen retrieval metrikleri yalnızca bu gözlenen ilişkilere göre yorumlanacaktır.

## 7. Karşılaştırılacak Modeller ve Ablation

| Deney | Girdi | Model | Amaç |
|---|---|---|---|
| Baseline-0 | Audio özet istatistikleri, BPM, lyrics, genre | Elle ağırlıklı cosine / uzaklık | Öğrenme olmadan referans |
| A | Segment-level audio | Temporal 1D-CNN | Audio-only kapasitesi |
| B | Audio + BPM | Temporal CNN + Tempo encoder | Tempo katkısı |
| C | Audio + BPM + Genre | Multi-branch fusion | Genre katkısı |
| D | Audio + BPM + Genre + Lyrics | Tam NeuroTune modeli | Lyrics katkısı, toplam etki |
| E (opsiyonel) | Tam çoklu girdi | CNN + GRU | Uzun dönem sıralamanın katkısı |

Tüm deneylerde aynı split, karşılaştırılabilir eğitim bütçesi, veri filtreleri ve retrieval havuzu kullanılacak.

### Ana metrikler

- **Recall@10:** Bilinen ilgili şarkıların ilk 10 içinde bulunması.
- **NDCG@10:** Sıralama kalitesi; ilgili şarkılar üst sıralarda mı?
- **MRR:** İlk ilgili şarkının sırası.
- **Triplet Accuracy (opsiyonel):** İnsan etiketli triplet verisi uyumlu şekilde kullanılabilirse ek dış değerlendirme.

Ek gözlemler: training time, embedding üretim süresi, sorgu gecikmesi, model boyutu.

## 8. Arama (Inference)

1. Eğitim tamamlandıktan sonra her şarkının embedding'ini **bir kez** hesapla.
2. 64D embedding'leri `float32` olarak sakla; L2 normalization uygula.
3. Bir sorgu geldiğinde sorgu embedding'ini elde et.
4. Vektörler arası dot product / cosine similarity ile Top-K seç.
5. Sorgu şarkısını kendi sonuçlarından çıkar.

Örneğin 10.000 şarkı ve 64D `float32` embedding için ham matris boyutu yaklaşık **2,56 MB**'tır (metadata ve indeks hariç). Başlangıç aşamasında NumPy ile tam cosine taraması yeterlidir; FAISS/HNSW zorunlu değildir.

Eğitim sırasında pair/triplet kullanılacak, ancak **sorgu anında her şarkı çifti ayrı neural network'e gönderilmeyecek**.

## 9. Teknoloji ve Kapsam

- **Dil:** Python.
- **Model:** PyTorch.
- **Veri:** NumPy, pandas, h5py, SciPy, scikit-learn.
- **Kayıtlar:** Parquet/CSV + NPZ/HDF5 + JSON manifest.
- **Retrieval:** Başlangıçta NumPy; ölçek gerektiğinde FAISS.
- **Arayüz:** Akademik deneyleri tamamladıktan sonra opsiyonel basit demo.

### Kapsam dışı (v1)

- Ham ses dalgasından uçtan uca büyük model eğitimi.
- Transformer veya büyük pretrained audio/language modelleri.
- Tam şarkı sözlerinin telifli metin olarak yeniden dağıtılması.
- Kullanıcı geçmişinden collaborative filtering.
- Milyon şarkılık dağıtık arama altyapısı.

## 10. Uygulama Aşamaları

| Aşama | Beklenen çıktı | Başarı koşulu |
|---|---|---|
| M0 - Veri uygunluğu | Kaynak eşleşme ve pozitif ilişki istatistikleri | Yeterli, ölçülmüş veri kesişimi |
| M1 - Audio baseline | Segment yükleyici + 1D-CNN | Modelin veri üzerinde eğitilebilmesi |
| M2 - Metric learning | Triplet sampling + loss | Loss ve retrieval metrikleri izlenebilir |
| M3 - Multimodal fusion | BPM, lyrics, genre encoder'ları | Tam model ve ablation çalışmaları |
| M4 - Değerlendirme | Recall@10, NDCG@10, MRR, tekrarlı seed'ler | Güvenilir karşılaştırma tabloları |
| M5 - Rapor ve demo | Final rapor + kaynak kod + opsiyonel UI | Tekrarlanabilir teslim paketi |

## 11. Kesinleştirilen Kararlar ve Açık Riskler

### Kararlar

- Audio time-series, modelin ana girdisi olacak.
- BPM, lyrics ve genre mutlaka ek girdi olarak hedeflenecek.
- V1 audio encoder: küçük **1D-CNN**.
- Mimari: ayrı encoder'lar, fusion MLP ve **64D normalize embedding**.
- Öğrenme: **Triplet metric learning**, paylaşılan encoder ağırlıkları.
- Retrieval: Önceden hesaplanan embedding'lerde **cosine similarity / Top-K**.
- İlk veri denemesi: **MSD 10K**.

### Açık riskler

1. Dört kaynağın aynı şarkılarda yeterli ölçüde örtüşmemesi.
2. Benzerlik etiketlerinin az, seyrek veya taraflı olması.
3. Last.fm benzerlik etiketleri ile genre kaynakları arasında döngüsel etiket etkisi.
4. Düzensiz segment zamanlarının yanlış yeniden örneklenmesi.
5. Çoklu modalitenin tek bir baskın modaliteye indirgenmesi; ablation ile izlenmeli.
6. Güçlü retrieval metrikleri elde edilse bile gerçek kullanıcı algısının ayrıca doğrulanmamış olması.

## 12. Sonuç

**NeuroTune v1 için seçilen ana mimari:**

> **MSD temporal pitch/timbre/loudness features → lightweight 1D-CNN → audio embedding; BPM + musiXmatch lyrics + tagtraum genre → dedicated encoders; fusion → 64D L2-normalized song embedding; shared-weight triplet training; cosine Top-K retrieval.**

**Bir sonraki uygulanacak görev:** M0 veri kesişimi ve pozitif benzerlik ilişkileri denetimi. Bu ölçümler, veri kaynağını ve eğitim örneklemesini kesinleştirecek; seçilen araştırma konusu ve genel mimari değişmeyecek.

---

## Referanslar

1. Lee vd. (2020), *Disentangled Multidimensional Metric Learning for Music Similarity*, ICASSP. https://doi.org/10.1109/ICASSP40776.2020.9053442
2. Ribecky, Abeßer ve Lukashevich (2022), *Multi-Input Architecture and Disentangled Representation Learning for Multi-Dimensional Modeling of Music Similarity*. https://arxiv.org/abs/2111.01710
3. Tran, Ngo ve d'Avila Garcez, *Probabilistic Approaches for Music Similarity Using Restricted Boltzmann Machines*, Neural Computing and Applications. https://doi.org/10.1007/s00521-019-04106-y
4. Sheikh Fathollahi ve Razzazi (2021), *Music Similarity Measurement and Recommendation System Using Convolutional Neural Networks*. https://doi.org/10.1007/s13735-021-00206-5
5. Million Song Dataset: https://millionsongdataset.com/
6. MSD musiXmatch dataset: https://millionsongdataset.com/musixmatch/
7. MSD Last.fm dataset: https://millionsongdataset.com/lastfm/
8. tagtraum genre annotations: https://www.tagtraum.com/msd_genre_datasets.html
