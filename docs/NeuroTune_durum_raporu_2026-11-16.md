# NeuroTune Durum Raporu

**İlk oluşturulma tarihi:** 16 Kasım 2026  
**Son güncelleme:** 17 Kasım 2026  
**Kapsam:** Veri hazırlığı, modeller, deneyler, doğrulama ve açık riskler

## Kısa özet

NeuroTune, MSD ses-analiz özellikleri ve Last.fm benzer parça bağlantılarıyla müzik benzerliği/retrieval deneyleri yapan bir ders projesidir. Veri denetimi, özellik çıkarımı, train-only dönüşümler, triplet eğitim, artist-grouped değerlendirme ve sonuç raporlama hatları tamamlandı.

En güncel çalışma, 13.822 parçalık A–D kataloğunda yapılan **P0 validation-only A/B/C/D ablation'ıdır**. A/B/D yeniden eğitildi; C checkpoint'leri read-only validation referansı olarak kullanıldı. D'nin öğrenilmiş model ortalaması C'nin biraz üzerinde, ancak eşleştirilmiş cosine baseline D modelini dört metriğin tamamında geçti. Checkpoint seçimi validation NDCG'ye dayandığından bu sonuçlar yalnızca geliştirme verisidir. Önceden incelenmiş test cohort'u bu çalışmada değerlendirilmedi.

## Mevcut durum

| Alan | Durum |
| --- | --- |
| Veri hattı | Resmî MSD 10K subset’i ve ayrı A/B, A–D pilot alanları hazır; kaynak ve çıktı hash’leri raporlanıyor. |
| En güncel veri | A–D pilotunda 13.822 cache-valid track; ilk 30 saniyeden 26 kanallı, normalizasyon uygulanmamış ragged ses girdisi. |
| En güncel model çalışması | P0 A/B/C/D ablation: A/B/D yeni eğitildi; C read-only validation referansı; D lyrics kullandı. |
| En güncel değerlendirme | 3-fold × 3-seed validation-only karşılaştırma; 36 model ve 12 cosine kaydı; test cohort'u kullanılmadı. |
| Önceki çıktılar | B0, N0, M1/M2, audio-only CV, A/B multimodal pilotu ve resmî 10K CV ayrı namespace’lerde korunuyor. |
| Genel sonuç | D, öğrenilmiş C'yi validation ortalamasında az farkla geçti; matched D cosine dört metriğin tamamında D modelinden yüksek. Üstünlük/genelleme sonucu yok. |

## Veri ve split

### Resmî 10K MSD subset’i

- 10.000 parçalık subset üzerinde tam ses + BPM + lyrics + tür kesişimi 939 parçadır.
- İlk B0/N0/M2 deneyleri küçük bu katalogla ve çok seyrek test grafiğiyle yapıldı; örneğin M1 test galerisi 89 parça ve 11 gözlenmiş kenar içeriyordu.
- Ayrı audio-only deneyinde duplicate filtrelemesinden sonra 9.939 MSD parçası kullanılabildi; bu yolun girdileri yalnızca MSD ses özellikleriydi, Last.fm bağlantıları ise denetim etiketi olarak kaldı.

### A/B ve A–D genişletmeleri

- A/B Hugging Face arşivleri 6.967 BPM + lyrics + tür adayı sağladı; bu üçüncü taraf yükleme proje kullanımı için kullanıcı tarafından onaylandı. Kamuya açık lisans ve kaynak/checksum bilgisi bağımsız olarak doğrulanmadı.
- A–D birlikte denetimde 154.801 ses kaydı incelendi; duplicate ve geçerlilik kurallarından sonra 13.822 tam-modlu aday kaldı. İlk 30 saniyelik 26-kanallı cache 1.430.546 sonlu float32 segment içeriyor; cache hazırlığında sıfır parça reddedildi.
- A–D’de train/test kaynakları arasındaki 1.204 yinelenen undirected bağlantı train etiketlerinden çıkarıldı. Son katalog kenarları: 67.657 Last.fm train-source ve 8.033 resmî test-source bağlantısı.
- A/B döneminden kalan 1.940 test-source kenarı yeni train/validation/test hedeflerinden çıkarıldı; yalnızca bilinen pozitifleri negatif örneklemeden hariç tutmak için kullanıldı.
- Yeni test artist’ları A/B kataloğunda bulunmayan artist gruplarından seçildi. Eski A/B artist’ları train/validation’da kalabilir, yeni test fold’larına giremez.

### En güncel üç fold

Her fold, A/B kataloğunda olmayan 588 test artist’ı içeriyor. Cross-partition kenarlar hedeflerden çıkarıldı. Train/validation kenarları Last.fm train arşivinden; test kaynakları ise iki ayrı dosyadan geliyor.

| Fold | Train / validation / test parça | Train-source train / val / test kenarı | Resmî test-source kenarı | Test sorgusu: train-source / resmî |
| --- | ---: | ---: | ---: | ---: |
| 0 | 11.640 / 1.377 / 805 | 49.266 / 811 / 183 | 50 | 260 / 73 |
| 1 | 11.693 / 1.335 / 794 | 48.523 / 835 / 221 | 30 | 299 / 50 |
| 2 | 11.669 / 1.381 / 772 | 47.992 / 831 / 235 | 45 | 273 / 69 |

## P0 mimarisi ve eğitim protokolü

- Girdi: ilk 30 saniyeden ragged `T×26` ses dizisi; varyanta göre BPM, genre ve musiXmatch lyrics.
- Ses dalı: masked `26→32→64→64` Conv1D ve masked temporal mean pooling; auxiliary dallar BPM `1→8`, genre `G→16` ve D'de lyrics `5000→32`; C fusion `88→128→64`, D fusion `120→128→64`, L2-normalize edilmiş embedding.
- Amaç: cosine triplet-margin loss, margin `0.2`; Adam `0.001`; batch size `64`; CPU; `50` epoch.
- Fold başına ses/BPM normalizasyonu, genre sözlüğü ve lyrics TF-IDF/IDF dönüşümü yalnızca train track’lerinde fit ediliyor.
- Train ve validation hedefleri yalnızca Last.fm train-source kenarları. Negatif örnekler uniform seçiliyor ve aday havuzundaki tüm bilinen pozitifler (validation/test-source dahil) negatiflerden çıkarılıyor. Bilinmeyen bağlantılar yine de doğrulanmış negatif değildir.
- Checkpoint seçimi validation NDCG@10, eşitlikte daha düşük validation loss, sonra daha erken epoch ile yapılıyor. C eski checkpoint'lerinden read-only alınırken A/B/D yeniden eğitildi.
- P0 runner yalnızca train/validation feature, embedding, relevance ve metrik üretti. Test-source pozitifleri sadece bilinen pozitifleri negatif örneklemeden çıkarmak için kullanıldı; train/validation hedefi olmadı.
- Sonuçlar `results/neurotune_v1_full_modal_ablation/` altında: 27 yeni checkpoint, dokuz paylaşılan schedule, 1.800 C+A/B/D eğri satırı. Mevcut test cohort'u değerlendirilmedi.

## P0 validation-only A/B/C/D ablation (17 Kasım 2026)

13.822 katalog track'i katı MXM coverage preflight'ından geçti; loader eksik ID bulursa koşuyu durduruyor. Rapor `empty_document_count: 0` kaydediyor, ancak ayrıca coverage sayacı yayımlamıyor. C'nin dokuz önceki validation seçimi read-only kullanıldı; A/B/D için 27 yeni koşu yapıldı. Her model ortalaması dokuz fold-seed, her cosine ortalaması üç fold gözlemidir.

| Model | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| A — audio | 0.048795 | 0.088177 | 0.034365 | 0.030221 |
| B — audio+BPM | 0.043948 | 0.078437 | 0.028857 | 0.026365 |
| C — audio+BPM+genre (read-only) | 0.118594 | 0.193172 | 0.069910 | 0.068638 |
| D — audio+BPM+genre+lyrics | 0.119240 | 0.194019 | 0.072913 | 0.071530 |

| Matched cosine baseline | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| A — audio | 0.041842 | 0.075427 | 0.031908 | 0.027089 |
| B — audio+BPM | 0.041812 | 0.080982 | 0.035022 | 0.028011 |
| C — audio+BPM+genre | 0.109255 | 0.190992 | 0.078063 | 0.068781 |
| D — audio+BPM+genre+lyrics | 0.121607 | 0.204624 | 0.087984 | 0.077921 |

D learned-model ortalaması C'den dört metriğin tamamında az miktarda yüksek; bu, lyrics'in bağımsız katkısını kanıtlamaz. D cosine baseline D modelini dört metriğin tamamında geçti. C modeli kendi cosine baseline'ını Recall@10 ve HitRate@10'da geçerken MRR@10 ve NDCG@10'da az farkla geride kaldı; A/B sonuçları daha düşüktü. Validation NDCG checkpoint seçimi için kullanıldığı, fold/seed gözlemleri bağımsız olmadığı ve Last.fm'de bulunmayan çiftler doğrulanmış negatif sayılmadığı için bu sonuçlar geliştirme kanıtıdır; genelleme veya üstünlük iddiası desteklemez.

Yeni A/B/D checkpoint seçili epoch'ları, seed sırası 684/685/686: A fold 0 `21/49/3`, fold 1 `33/13/18`, fold 2 `38/6/16`; B fold 0 `18/34/24`, fold 1 `26/24/34`, fold 2 `18/19/12`; D fold 0 `44/2/37`, fold 1 `25/1/6`, fold 2 `20/42/2`. Bu geniş yayılım, özellikle D'deki epoch `1–2` seçimleri, validation sonuçlarının fold/seed'e duyarlı olduğunu gösterir; test sonucuyla çözülmeye çalışılmamalıdır.

Runner raporu `status: complete` ve bağımsız artifact doğrulaması 56 checkpoint/schedule/curve hash'inin raporla eşleştiğini, C referans dosyalarının değişmediğini gösterdi. Koşu yaklaşık 18 saat 45 dakika sürdü; host sürecinin exit code'u geri alınamadı, bu nedenle completion manifesti ve hash kontrolleri doğrulama kanıtıdır.

## Önceki C CV test sonuçları (tarihsel kayıt; P0'da kullanılmadı)

Değerler ortalama ± popülasyon standart sapmasıdır. Model satırları dokuz fold-seed gözlemi, cosine satırları üç fold gözlemidir.

| Test etiketi kaynağı | Yöntem | Gözlem | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Last.fm train arşivinden ayrılan kenarlar | C | 9 | 0.1681 ± 0.0274 | 0.2232 ± 0.0347 | 0.0769 ± 0.0186 | 0.0895 ± 0.0178 |
| Last.fm train arşivinden ayrılan kenarlar | C-matched cosine | 3 | 0.1510 ± 0.0248 | 0.2022 ± 0.0309 | 0.0682 ± 0.0089 | 0.0787 ± 0.0126 |
| Resmî Last.fm test arşivi | C | 9 | 0.1521 ± 0.0556 | 0.1802 ± 0.0699 | 0.0560 ± 0.0252 | 0.0760 ± 0.0292 |
| Resmî Last.fm test arşivi | C-matched cosine | 3 | 0.1649 ± 0.0716 | 0.1970 ± 0.0854 | 0.0801 ± 0.0449 | 0.0920 ± 0.0446 |

C, train-source etiketlerinde ortalamada dört metriğin tamamında cosine’ın önünde; resmî test-source etiketlerinde ise dördünde de gerisinde. Fold’lar aynı Last.fm kaynak havuzlarını yeniden kullandığından, dokuz model koşusu dokuz bağımsız örnek sayılmaz. Resmî test-source tarafında fold başına 30–50 kenar bulunuyor. Bu farklar istatistiksel üstünlük ya da geniş kullanıcı kitlesine genelleme sonucu olarak yorumlanmamalı.

Seçilen epoch’lar seed sırası 684/685/686 olacak şekilde fold 0 için `24/12/25`, fold 1 için `22/4/16`, fold 2 için `4/4/5` oldu. Ayrıntılı fold ve seed ölçümleri, hash’ler ve çalışma ortamı `results/neurotune_v1_abcd_cv/experiment.json` içindedir.

## Önceki deneylerin özeti

Bu deneyler farklı katalog, split, hedef ve değerlendirme protokolleri kullandı; metrikleri doğrudan birbiriyle kıyaslanmamalıdır.

| Deney | Kapsam | Önemli sonuç / mevcut yorum |
| --- | --- | --- |
| M0 / B0 | Resmî 10K; 939 tam-modlu parça; metadata retrieval | B0 Recall@10 ve HitRate@10 `0.45`; 89 parçalık test galerisi ve yalnızca 11 gözlenmiş test kenarı nedeniyle keşif amaçlı. |
| N0 | BPM + lyrics + tür tabanlı Siamese model | Ortalama Recall@10 `0.4917`; küçük test grafiği ve train/validation ayrışması overfitting riski gösterdi. Güvenilir model üstünlüğü değil. |
| M1 / M2 | 939 parçalık temporal ses yolu; M2 ses + metadata | M2 ortalama Recall@10 `0.4667`; testte 11 gözlenmiş kenar var. Model ayrı, keşif amaçlı prototip olarak tutuluyor. |
| MSD audio-only CV | 9.939 parça; yalnızca MSD ses girdisi, Last.fm supervision | Model ortalama Recall@10 `0.0133`; sonradan ölçülen mean-pooled cosine `0.0394`. Farklı fold/observation yapıları nedeniyle tanısal, kesin üstünlük kanıtı değil. |
| A/B multimodal v1 | 6.967 parça; artist-disjoint tek test cohort’u | C: `0.1887/0.2629/0.0912/0.1031`; cosine: `0.1803/0.2528/0.1105/0.1120` (Recall/HitRate/MRR/NDCG). C Recall/HitRate’ta az farkla önde, MRR/NDCG’de gerideydi. |
| Resmî 10K C CV | 939 parça; üç artist fold | C ortalaması `0.2618/0.3270/0.1202/0.1440`; C-matched cosine `0.2322/0.2910/0.1106/0.1307`. Küçük ve aynı kaynak havuzuna dayalı robustness kontrolü. |
| A/B validation score fusion | Validation-only, test kullanılmadı | En iyi validation NDCG `λ=1` (C tek başına); cosine eklemek validation sonucunu iyileştirmedi. Aynı validation ile checkpoint seçildiği için bağımsız kanıt değil. |
| Full-modal P0 ablation | A/B/D yeni eğitildi; C read-only; validation-only | D learned model C'den az farkla yüksek; D cosine baseline D modelini dört metriğin tamamında geçti. Test cohort'u değerlendirilmedi. |

## Çıktılar ve yeniden üretim

En güncel validation-only çalışma:

- Rapor: `results/neurotune_v1_full_modal_ablation/experiment.json`
- Eğitim eğrileri: `results/neurotune_v1_full_modal_ablation/training_curves.csv` (1.800 satır)
- Triplet schedule kopyaları: `results/neurotune_v1_full_modal_ablation/triplet_schedules/` (9 dosya)
- Yeni A/B/D checkpoint'leri: `results/neurotune_v1_full_modal_ablation/checkpoints/` (27 dosya)
- C read-only referansları: `results/neurotune_v1_abcd_cv/` (değiştirilmedi)

Proje kökünden çalıştırma:

```bash
.venv/bin/python -m scripts.run_neurotune_v1_full_modal_ablation
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/python -m pip check
```

Eski C CV komutu `scripts.run_neurotune_v1_abcd_cv` ayrı ve tamamlanmış deneyin yeniden üretimi içindir; mevcut C namespace'i P0 tarafından açılmadı veya değiştirilmedi.

## Önceki C koşusunun doğrulaması

- Sabit C CV koşusu yaklaşık 18.165 saniye (5 saat 3 dakika) sürdü.
- Çıktı raporundaki input, code, manifest, cache, transform, schedule ve checkpoint hash’leri yeniden hesaplanıp eşleştirildi.
- Dokuz schedule’daki toplam 43.749.162 triplet satırı tarandı: pozitif hedefler doğru partition’dan geliyor ve hiçbir negatif bilinen pozitif bağlantı değil.
- Çalışma öncesi snapshot’ta bulunan 9.465 eski sonuç/cache dosyası byte-for-byte değişmeden kaldı.
- Son doğrulama: 396 test geçti; `compileall` başarılı; `pip check`: `No broken requirements found.`
- Çalışma alanı Git deposu değil; bu nedenle commit oluşturulmadı.

## P0 artifact doğrulaması

- Read-only fold-0 bellek ön kontrolü 881.29 MiB peak RSS ölçtü; aynı anda 10,213 MiB kullanılabilir bellek bildirildi.
- P0 manifestinde `status: complete`; çıktı toplamı 38 dosya / 222.128.880 byte: 27 checkpoint, 9 schedule, eğri CSV ve JSON raporu.
- 36 model validation kaydı, 12 cosine kaydı ve 1.800 curve satırı doğrulandı; her varyantta 450 satır, her fold/seed grubunda 50 epoch var.
- Yeni checkpoint/schedule hash'leri ve C checkpoint/schedule/curve referans hash'leri raporla eşleşti; C namespace'i değişmedi.
- 36 checkpoint'in variant/fold/seed/epoch, validation selection metrics ve transform/schedule hash metadata'sı raporla eşleşti.
- Full suite: 423 test geçti; `compileall` başarılı; `pip check`: `No broken requirements found.`
- Suite'in açığa çıkardığı eski sentetik MXM örneği `MXM1` yazıyordu; resmi dosya biçimi ikinci alanı sayısal MXM track ID olarak tanımlıyor. Fixture `12345` olacak şekilde düzeltildi; parser kodu değiştirilmedi.
- Ana eğitim sürecinin OS exit code'u host tarafından geri verilmedi. Buna rağmen completion manifesti, bağımsız hash denetimi ve yukarıdaki ayrı doğrulama komutları geçti.

## Açık riskler ve sonraki adımlar

1. Güncel A–D test cohort’u veya resmî 10K CV test sonuçları üzerinde yeni model/hiperparametre ayarı yapma. Yeni bir karar gerekiyorsa yalnızca daha önce belirlenmiş validation protokolünü kullan.
2. Daha güçlü performans iddiası için bağımsız, artist-held-out ve lisansı doğrulanmış yeni bir test kohortu oluştur; etiket kaynaklarını model seçimi ve nihai değerlendirme arasında ayır.
3. A–D arşivlerinin kamuya açık lisans ve kaynak provenansını yayımlama/dağıtım öncesinde ayrıca doğrula. Kullanıcının proje içi kullanım onayı kamu lisansı doğrulaması anlamına gelmez.
4. Ders teslimi öncesi Python kullanımının, beklenen rapor biçiminin ve proje dilinin ders gereksinimleriyle uyumunu doğrula.

## İlgili belgeler

- Proje özeti ve kullanıcıya dönük sonuçlar: `README.md`
- A–D C CV tasarımı: `docs/plans/2026-11-15-neurotune-abcd-c-cv-training-design.md`
- A–D C CV uygulama/ doğrulama planı: `docs/plans/2026-11-15-neurotune-abcd-c-cv-training-implementation.md`
- Mevcut v1 mimari durumu: `docs/NeuroTune_v1_current_status_draft.md`
