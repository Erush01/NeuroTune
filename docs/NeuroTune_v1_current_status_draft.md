# NeuroTune v1 — Güncel Mimari ve Deney Durumu (Taslak)

**Ders:** CMP684 — Neural Networks  
**Proje:** Müzik benzerliği için şarkı embedding'leri  
**Belge durumu:** Mevcut kod ve deney raporlarıyla uzlaştırılmış taslak. Literatür atıflarının bibliyografik/teknik doğruluğu bu güncellemede yeniden doğrulanmamıştır.  
**Tarih:** 12 Kasım 2026

## 1. Amaç ve araştırma sorusu

NeuroTune, MSD şarkı analiz özelliklerinden bir embedding üretip cosine similarity ile benzer şarkı aramayı araştırır. Uzun vadeli hedef temporal audio özelliklerini merkeze alıp BPM, musiXmatch lyric BoW ve tagtraum genre ile tamamlamaktır. Ancak hedef mimari, mevcut prototipler ve ölçülmüş sonuçlar aynı şey değildir.

**Güncel araştırma sorusu:** Temporal MSD audio temsili retrieval'da ne kadar başarılıdır ve yardımcı BPM, lyrics, genre modaliteleri aynı katalog ve split üzerinde kontrollü ablation'larda ek değer sağlar mı?

Last.fm benzer-şarkı bağlantıları gözetim sinyalidir. Gözlenmeyen bir bağlantı doğrulanmış biçimde “benzer değil” anlamına gelmez. Tüm sonuçlar keşif amaçlıdır; mevcut test grafikleri güçlü genelleme veya istatistiksel üstünlük iddiası için yetersizdir.

## 2. Doğrulanmış veri kapsamı

### Multimodal 10K kesişimi

- Resmî MillionSongSubset'te 10.000 şarkı HDF5 kaydı vardır.
- BPM, musiXmatch lyrics, tagtraum CD2C genre ve benzerlik verilerinin kesişimi, duplicate filtrelemesi sonrasında 939 şarkıdır.
- Mevcut artist-disjoint split 756/94/89 parçadır; pozitif Last.fm bağlantıları train/validation/test için 416/6/11'dir. Bu küçük validation/test grafikleri nedeniyle sonuçlar keşif amaçlıdır.

### Geniş audio-only CV kataloğu

- Duplicate filtrelemesi sonrası 9.939 şarkı vardır. Bu deneyde BPM, lyrics ve genre model girdisi değildir.
- Deterministik artist-grouped üç fold 3.249/3.321/3.369 parça içerir.
- Fold-içi pozitif bağlantılar 1.065/965/981; fold'lar arası 5.198 bağlantı değerlendirmeden çıkarılmıştır.
- Her fold'un ayrı cache'i vardır ve normalization yalnızca diğer iki eğitim fold'unda fit edilir.

## 3. Uygulanan MSD temporal audio temsili

Bu, waveform veya log-mel spectrogram değildir; MSD HDF5 analiz özellikleridir. Preprocessing her parçayı 1.024 noktaya yeniden örnekleyerek `float32`, `(1024, 28)` dizi üretir:

| Kanallar | İçerik | Normalizasyon |
|---|---|---|
| 0–11 | Segment pitch (12) | Sürekli kanallar; yalnız train ID'lerinden fit edilir |
| 12–23 | Segment timbre (12) | Sürekli kanallar; yalnız train ID'lerinden fit edilir |
| 24 | Segment loudness (1) | Sürekli kanallar; yalnız train ID'lerinden fit edilir |
| 25–27 | Beat, bar, section event-count | Değişmeden bırakılır |

Başlangıç taslağındaki `T x 26`, ayrı duration kanalı ve masked pooling mevcut uygulamayla uyuşmuyor. Uygulamada açık segment-duration kanalı/masked pooling yok; continuous özellikler normalized-time grid'e interpolasyonla taşınır, olaylar grid üzerinde sayılır.

## 4. Önerilen hedef ile tamamlanmış modeller

| Çalışma | Kapsam ve girdiler | Model/eğitim | Durum |
|---|---|---|---|
| B0 | 939 parçalık kesişim; BPM, lyric TF-IDF, genre | Eşit ağırlıklı elle tasarlanmış özellikler + cosine | Tamamlandı; keşif baseline'ı |
| N0 | Aynı 939 parça; BPM, lyric TF-IDF, genre | 64D Siamese; contrastive margin loss | Tamamlandı; metadata-only prototip |
| M2 audio-first | Aynı 939 parça; temporal audio + BPM + lyrics + genre | CNN audio kolu, 16 sıralı pooled bin, 64D normalize embedding; contrastive loss | Tamamlandı; multimodal prototip |
| Audio-only | 9.939 parça; yalnız temporal MSD audio; Last.fm supervision | 32D Siamese CNN; contrastive loss; uniform rastgele gözlenmemiş negatifler | Tamamlandı; tek-split keşif deneyi |
| Audio-only 3-fold CV | 9.939 parça; yalnız temporal MSD audio | Aynı 32D model, üç artist-grouped fold ve fold'a özel train-normalized cache | Tamamlandı; keşif CV deneyi |

Başlangıç belgesindeki **64D audio encoder + BPM/lyrics/genre fusion + triplet loss** mevcut tek bir model/protokol değildir. M2, temporal audio ve auxiliary modaliteleri birleştiren 64D prototiptir fakat contrastive loss kullanır. Geniş CV modeli yalnızca audio kullanan 32D modeldir ve contrastive loss kullanır. Katalog/split'ler farklı olduğundan iki deneyin skorları doğrudan karşılaştırılmamalıdır.

## 5. Mevcut sonuçlar

### M2 multimodal — 939 parçalık katalog

89 parçalık test galerisi, 20 sorgu ve 11 gözlenen test bağlantısında üç seed ortalaması:

| Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
|---:|---:|---:|---:|
| 0.4667 | 0.4667 | 0.1529 | 0.2267 |

Eğitim kaybı azalırken validation kaybı 50. epoch'a doğru yükseldi. Altı validation ve 11 test bağlantısı güvenilir model seçimi/performans iddiası için yeterli değildir.

### Audio-only 3-fold CV ve post-hoc baselinelar

Üç yöntemin kıyası aynı held-out fold'lardadır, ancak gözlem sayısı farklıdır: mean-pooled cosine için 3 fold, öğrenilmiş model ve random için dokuz fold-seed ölçümü.

| Yöntem | Gözlem sayısı | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Mean-pooled audio cosine | 3 fold | 0.0394 ± 0.0064 | 0.0657 ± 0.0098 | 0.0271 ± 0.0068 | 0.0243 ± 0.0041 |
| Learned audio-only model | 9 fold-seed | 0.0133 ± 0.0042 | 0.0263 ± 0.0058 | 0.0083 ± 0.0022 | 0.0073 ± 0.0022 |
| Random top-10 ranking | 9 fold-seed | 0.0031 ± 0.0021 | 0.0058 ± 0.0033 | 0.0018 ± 0.0011 | 0.0016 ± 0.0010 |

± değerleri ilgili gözlemler üzerindeki sample standard deviation'dır; inferans aralığı değildir. Mean-pooled cosine bu post-hoc kıyasta dört agregat metriğin hepsinde öğrenilmiş audio-only modelden yüksek çıkmıştır. Bu, basit pooled temsilin bu deneyde güçlü bir tanı baseline'ı olduğunu gösterir; cosine'ın genelde daha iyi olduğunu veya modelin başka kataloglarda başarısız olacağını kanıtlamaz. Test fold'ları daha önce incelenmiştir; sonuçlar keşif amaçlıdır ve gözlenmeyen Last.fm bağlantıları doğrulanmış negatifler değildir.

Deney raporları: `results/baseline_b0.json`, `results/n0_experiment.json`, `results/m2_audio_experiment.json`, `results/msd_audio_only_experiment.json`, `results/msd_audio_only_cv_experiment.json`, `results/msd_audio_only_cv_baselines.json`.

## 6. Etiket ve eğitim sınırlamaları

Mevcut öğrenilmiş prototipler Siamese çiftler ve contrastive margin loss kullanır; başlangıç belgesindeki triplet loss tamamlanmış eğitim protokolü değildir. Audio-only deneyinde her pozitif için bir rastgele, gözlenmemiş çift negatif örneklenmiştir. Gözlenmemiş olması gerçek dissimilarity etiketi değildir ve false-negative riski taşır.

Last.fm seyrek davranışsal supervision sağlar, insan benzerliğinin eksiksiz ölçümü değildir. Retrieval metrikleri yalnızca gözlenen held-out bağlantılara göre okunmalıdır; kullanıcı beğenisi veya anlamsal benzerlikle özdeşleştirilmemelidir.

## 7. Sonraki adımlar

Post-hoc test sonuçlarını yeni model seçimi için kullanmayın. Yeni bir önceden tanımlanmış deneyde:

1. Pooled-cosine'ı sonraki deneylerde sabit bir referans baseline olarak kullanın.
2. Gözlenmemiş Last.fm çiftlerinin negatif sayılmasının etkisini araştırın; weak-negative/güvenilir negatif stratejilerini yalnızca eğitim fold'larının iç artist-disjoint validation'ında seçin.
3. Audio-only ve audio+BPM+lyrics+genre ablation'larını aynı parça kataloğu, aynı fold'lar ve aynı eğitim bütçesinde karşılaştırın; 939 parçalık multimodal kısıtı raporlayın.
4. Contrastive ve triplet objective karşılaştırılacaksa seçim için outer test fold'larını değil, iç validation'ı kullanın.
5. 28 kanal ve auxiliary modalite katkılarını kontrollü ablation'larla ölçün. Duration kanalı, GRU veya daha büyük encoder'ı yeni hipotez/test olmadan eklemeyin.

## 8. Başlangıç belgesindeki güncellenen iddialar

| Eski ifade | Güncel durum |
|---|---|
| “Veri kümesi ve sonuçlar henüz doğrulanmadı”; sıradaki iş M0 | M0, M1, M2, audio-only tek split ve CV tamamlandı; kapsam/limitler ölçüldü. |
| Bütün 10K için multimodal training yapılabilir izlenimi | Yardımcı kaynakların kesişimi 939; duplicate-filtered audio-only kapsam 9.939. |
| `T x 26`, duration kanalı, masked pooling | Uygulama `(1024,28)`; 25 continuous + 3 event-count kanal; explicit duration/masked pooling yok. |
| 64D multimodal triplet ağ tamamlanmış gibi | M2 64D multimodal contrastive; geniş CV 32D audio-only contrastive; triplet henüz uygulanmadı. |
| Elle ağırlıklı cosine tek baseline | B0 buna yakındır; ayrıca audio-only CV'de ayrı mean-pooled cosine ve random baselinelar vardır. |

## 9. Literatür notu

Kaynak listesi başlangıç taslağından taşınmıştır; bu uzlaştırmada bibliyografik metaveri, veri sürümleri veya makale mimarileri birincil kaynaklardan yeniden kontrol edilmedi. Akademik rapora eklemeden önce doğrulayın. Veri erişimi ve lisans koşulları da ilgili kaynakların güncel şartlarına göre kontrol edilmelidir.

1. Lee vd. (2020), *Disentangled Multidimensional Metric Learning for Music Similarity*, ICASSP. https://doi.org/10.1109/ICASSP40776.2020.9053442
2. Ribecky, Abeßer ve Lukashevich (2022), *Multi-Input Architecture and Disentangled Representation Learning for Multi-Dimensional Modeling of Music Similarity*. https://arxiv.org/abs/2111.01710
3. Tran, Ngo ve d'Avila Garcez, *Probabilistic Approaches for Music Similarity Using Restricted Boltzmann Machines*, Neural Computing and Applications. https://doi.org/10.1007/s00521-019-04106-y
4. Sheikh Fathollahi ve Razzazi (2021), *Music Similarity Measurement and Recommendation System Using Convolutional Neural Networks*. https://doi.org/10.1007/s13735-021-00206-5
5. MSD: https://millionsongdataset.com/ · musiXmatch: https://millionsongdataset.com/musixmatch/ · Last.fm: https://millionsongdataset.com/lastfm/ · tagtraum: https://www.tagtraum.com/msd_genre_datasets.html
