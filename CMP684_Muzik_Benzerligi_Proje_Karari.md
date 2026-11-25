# CMP684 Final Projesi: Mimari ve Kapsam Kararı

**Durum:** Konu ve temel mimari kararlaştırıldı  
**Proje türü:** Neural network tabanlı benzer şarkı bulma (music similarity retrieval)  
**Amaç:** Bir şarkının zamana bağlı müzikal özelliklerini temel alarak BPM, lyrics ve genre bilgileriyle zenginleştirilmiş bir embedding üretmek; benzer şarkıları bu embedding'ler üzerinden sıralamak.

## 1. Kesinleşen karar

**Time-series audio features sistemin ana girdisi olacak.** Şarkıların yalnızca BPM, söz ve tür gibi özet özellikleriyle yetinilmeyecek. Pitch, timbre, loudness ve ritmik/zamansal yapı gibi özelliklerin **zaman içindeki değişimi** neural network tarafından işlenecek.

- **Ana modalite:** Zaman serisi biçimindeki akustik özellikler.
- **Yardımcı modaliteler:** BPM (tempo), lyrics (şarkı sözlerinin sayısal temsili), genre (müzik türü).
- **Öğrenilen çıktı:** Her şarkı için sabit uzunluklu, karşılaştırılabilir bir embedding vektörü (başlangıç hedefi: **64 boyut**).
- **Eğitim:** Ağırlıkları paylaşılan iki encoder'lı **Siamese** yaklaşımı ve benzerlik odaklı loss.
- **Arama:** Önceden hesaplanmış embedding'ler üzerinde **cosine similarity** ile Top-K benzer şarkı önerisi.
- **İlk hedef:** Gereksiz büyük bir model yerine küçük ve ölçülebilir bir neural network sistemi.

> **Araştırma sorusu:** Zamana bağlı akustik özellikleri BPM, lyrics ve genre ile birleştirmek, benzer şarkı bulma başarısını yalnızca metadata ya da yalnızca time-series kullanan modellere göre artırıyor mu?

## 2. Önerilen model mimarisi

```text
                  ŞARKI
                    |
        +-----------+------------+------------+
        |           |            |            |
  Time-series     BPM          Lyrics        Genre
  Pitch/Timbre    Tempo        BoW/TF-IDF    One-hot
  Loudness/Beat   scalar       vector        vector
        |           |            |            |
   Temporal     Small MLP    Lyrics MLP    Genre MLP
   1D-CNN       encoder      encoder       encoder
        |           |            |            |
  64D audio      8D           32D          16D
        +-----------+------------+------------+
                    |
             Feature fusion
                    |
               Fusion MLP
                    |
       64D L2-normalized song embedding
                    |
          Cosine similarity / Top-K
```

Boyutlar **başlangıç konfigürasyonudur**, deneylere göre değiştirilebilir. Ana mimari kararı, bir **temporal audio encoder** ve yardımcı metadata encoder'larını birleştirmektir.

### Time-series encoder

İlk tercih: **hafif 1D-CNN + temporal pooling**.

- `segments_pitches`: `T x 12` nota sınıfı (chroma) özellikleri.
- `segments_timbre`: `T x 12` tını özellikleri.
- `segments_loudness_max`: `T x 1` ses şiddeti özelliği.
- `segments_start`: segment zamanları ve süreleri için referans.
- `beats_start`, `bars_start`, `sections_start`: ritmik ve yapısal zaman bilgisi; uygun hizalama/özellik üretimiyle kullanılacak.

`T` şarkıya göre değiştiği için segmentleri belirli bir zaman çözünürlüğüne yeniden örnekleme, padding/masking ve/veya maskeli temporal pooling uygulanacak. **Segment sırası korunacak**; yalnızca ortalama ve standart sapma çıkarmak ana modelin yerine geçmeyecek.

**Not:** 1D-CNN kullanımı modelin zamansal derin öğrenme bileşeni içerdiği anlamına gelir. Önceki yalnızca sığ MLP planından bilinçli bir değişikliktir. Dersin neural network kapsamıyla uyumlu, küçük bir CNN hedeflenmektedir.

### Yardımcı encoder'lar

| Girdi | Temsil | Örnek encoder |
|---|---|---|
| BPM | Normalize edilmiş tempo; gerekirse yarım/çift tempo ilişkileri | Küçük MLP, 8D |
| Lyrics | musiXmatch bag-of-words; train verisiyle fit edilmiş TF-IDF seçeneği | MLP, 32D |
| Genre | Tekli one-hot veya uygun veri varsa multi-hot | MLP, 16D |

Lyrics verisi tam metin değilse buna uygun yöntem kullanılacak. **musiXmatch'in MSD için yayımladığı veri bag-of-words'tür; cümle sırası ve tam semantik içerik sunmaz.**

## 3. Veri stratejisi

### Ana veri kaynağı

**Million Song Dataset (MSD)**. Time-series özellikler için yalnızca `msd_summary_file.h5` yeterli değildir; şarkıların **ayrıntılı HDF5 analiz dosyaları** gereklidir.

**İlk aşamada resmi MSD 10K Subset ile prototip kurulacak.** Bu subset zamana bağlı analiz özelliklerini içerir, fakat gerçek `.mp3` / `.wav` ses kayıtlarını içermez.

### Diğer veri kaynakları

| Kaynak | İşlev |
|---|---|
| MSD 10K / detaylı MSD HDF5 | Pitch, timbre, loudness, beats, segment timing; tempo ve metadata |
| musiXmatch MSD lyrics | Şarkı sözü bag-of-words özellikleri |
| tagtraum / uygun tür etiketi verisi | Genre bilgisi |
| Last.fm similarity verisi | Pozitif benzer şarkı ilişkileri ve varsa ilişki ağırlıkları |

Kaynaklar **`track_id` üzerinden** birleştirilecek. İsim benzerliğine dayalı otomatik eşleştirme ana yöntem olmayacak.

**Henüz doğrulanmamış nokta:** Dört kaynağın 10K subset üzerindeki ortak şarkı sayısı ve bu şarkılar arasında kalan pozitif benzerlik çifti sayısı. Veri hazırlamanın ilk ölçülecek çıktıları bunlardır.

Yeterli eşleşme çıkmazsa mimari değiştirilmeyecek; kullanılabilir kapsamı büyütmek için daha geniş MSD içinden ilgili şarkıların ayrıntılı HDF5 dosyaları alınacak.

### Beklenen veri çıktıları

```text
data/processed/
  songs.parquet           # ID, artist, title, BPM, genre, feature referansları
  audio_sequences/        # Şarkı bazında zaman serisi tensörleri
  lyrics.npz              # Sparse lyrics özellikleri
  similar_pairs.parquet   # Last.fm benzerlik ilişkileri
  splits.parquet          # Train/validation/test kayıtları
  metadata.json           # Kaynaklar, filtreler, eşleşme istatistikleri
```

## 4. Eğitim yaklaşımı

İki şarkı, **aynı ağırlıkları paylaşan encoder** ile ayrı ayrı işlenir:

\[
z_A = f_\theta(X_A, b_A, l_A, g_A), \qquad
z_B = f_\theta(X_B, b_B, l_B, g_B).
\]

Burada:

- `X`: zaman serisi akustik özellikleri,
- `b`: BPM,
- `l`: lyrics,
- `g`: genre,
- `z`: L2-normalize şarkı embedding'i.

**Pozitif çiftler** Last.fm benzer şarkı ilişkilerinden, **negatif çiftler** kontrollü sampling ile oluşturulacak. Last.fm'de ilişkisi olmayan her şarkı çifti kesin negatif kabul edilmeyecek.

İlk loss tercihi **contrastive loss** (veya eğitim performansına göre triplet loss). Tüm olası şarkı çiftleri eğitimde üretilmeyecek; yalnızca seçilmiş çiftler/minibatch'ler kullanılacak.

Eğitim tamamlandıktan sonra şarkı embedding'leri **bir kez hesaplanıp saklanacak**. Her arama sırasında binlerce çift tekrar neural network'ten geçirilmeyecek.

## 5. Deney planı

Tüm deneylerde aynı veri bölmesi, aynı değerlendirme sorguları ve karşılaştırılabilir eğitim bütçesi kullanılacak.

| Deney | Özellikler | Amaç |
|---|---|---|
| Baseline | Elle tanımlanmış/normalize edilmiş özellik uzaklığı | Neural olmayan referans |
| A | BPM + Lyrics + Genre | Metadata-only performansı |
| B | **Time-series audio only** | Ana modalitenin tek başına katkısı |
| C | Time-series + BPM | Tempo eklemenin katkısı |
| D | Time-series + BPM + Genre | Tür bilgisinin katkısı |
| E | **Time-series + BPM + Genre + Lyrics** | Tam model |

**Temel metrikler:** `Recall@10`, `NDCG@10`, `MRR` (yeterli ilgili şarkı etiketi varsa `Precision@10`). Özelliklerin etkisi için ablation study yapılacak.

### Deney güvenilirliği

- Train/validation/test ayrımı yapılmadan veri istatistikleri fit edilmeyecek.
- Aynı şarkıların ya da eşleştirilmiş kopyaların train ve test arasında bilgi sızdırması engellenecek.
- En azından **görülmemiş sorgu şarkıları** ile değerlendirme yapılacak; mümkünse sanatçı bazlı ayrım ayrıca raporlanacak.
- Sadece aynı bölmenin iki tarafına rastgele çift dağıtmak yeterli sayılmayacak.
- Kullanılabilir test sorgusu ve ilgili şarkı sayısı açıkça belirtilecek.
- Erişilemeyen özellikler, eksik veri oranları ve geçerli çift sayıları saklanmayacak; raporda açıklanacak.

## 6. Geliştirme aşamaları

1. **M0 - Dataset audit:** MSD 10K detaylı HDF5, lyrics, genre ve Last.fm örtüşmesini ölç. Geçerli şarkı, pozitif çift ve sorgu sayısını raporla.
2. **M1 - Audio preprocessing:** Zaman serilerini yükle, hizala/yeniden örnekle, normalleştir ve önbellekle. Birkaç şarkı üzerinde shape/timing testleri yaz.
3. **M2 - Baseline:** Metadata ve basit audio istatistikleriyle nearest-neighbor sonuçlarını hesapla.
4. **M3 - Temporal encoder:** 1D-CNN + pooling ile audio-only Siamese modeli eğit.
5. **M4 - Multimodal fusion:** BPM, genre ve lyrics encoder'larını ekle; tam model eğit.
6. **M5 - Evaluation:** Ortak test sorgularında ablation, retrieval metrikleri, hata analizi ve süre/bellek ölçümleri yap.
7. **M6 - Demo ve rapor:** Seçilen şarkı için Top-10 öneri gösteren basit demo ve CMP684 akademik raporu hazırla.

## 7. Kapsam dışı / ertelenen işler

- Tüm milyon şarkı üzerinde ilk günden eğitim.
- Ham ses dalgasından büyük bir end-to-end model eğitmek.
- Transformer veya ağır pretrained audio-language modellerini zorunlu tutmak.
- Gerçek zamanlı müzik tanıma, streaming servis entegrasyonu veya üretim seviyesinde öneri servisi.

## 8. Nihai kararın özeti

**Kabul edilen sistem:**

> MSD'nin zaman serisi biçimindeki akustik özelliklerini temel alan; BPM, lyrics ve genre bilgisini yardımcı encoder'larla birleştiren; Siamese neural network olarak eğitilen; her şarkı için sabit boyutlu embedding üreten ve cosine similarity ile benzer şarkıları sıralayan bir müzik retrieval sistemi.

**Karar verilmiş olan:** Audio-first multimodal mimari, temporal encoder gerekliliği, Siamese eğitim ve embedding tabanlı arama.

**M0 sonrasında belirlenecek olan:** Nihai kullanılabilir şarkı sayısı, veri kaynakları arasındaki kesişim, tam giriş boyutları, örnekleme stratejisi ve eğitim hiperparametreleri.

---

**Kaynak başlangıç noktaları:** [Million Song Dataset](https://millionsongdataset.com/), [MSD additional datasets](https://millionsongdataset.com/pages/additional-datasets/), [tagtraum genre datasets](https://www.tagtraum.com/msd_genre_datasets.html).
