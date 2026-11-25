# NeuroTune D Modeli: Uygulama ve Validation Sonuçları

**Tarih:** 17 Kasım 2026  
**Durum:** D varyantı eğitildi; yalnızca geliştirme/validation değerlendirmesi yapıldı.  
**Ana çıktı:** `results/neurotune_v1_full_modal_ablation/experiment.json`

**Sonraki tanısal analiz:** `docs/NeuroTune_D_Modalite_Duyarliligi_2026-11-17.md` — sabit D checkpoint'lerinde, test seti kullanılmadan, her girdiyi ayrı sıfırlayan validation duyarlılığı.

## Kısa cevap: Yeni bağımsız test neden gerekli?

D modelinin **uygulanması ve eğitilmesi tamamlandı**. Yeni test, modelin çalıştığını göstermek için değil, daha önce model seçmekte kullanılmamış şarkı ve insan/başka kaynak etiketleri üzerinde de başarılı olup olmadığını **tarafsız ölçebilmek** için gerekli. Bu çalışmada validation NDCG@10 her koşunun checkpoint'ini seçti; aynı validation üzerinde A/B/C/D karşılaştırması yapıldı. Dolayısıyla aşağıdaki sayılar model geliştirmeye yardımcı olur, bağımsız başarı veya genelleme kanıtı değildir. Önceden incelenmiş A–D test cohort'u bu ablation için yeniden kullanılmadı. Ders projesinde D'yi tamamlanmış deneysel prototip olarak sunabiliriz; “D daha iyi genelliyor” iddiasını bu rakamlarla kuramayız.

## Girdi ve mimari

- Katalog: A–D pilotundaki **13.822** parça. Ses girdisi her parça için ilk 30 saniyenin normalize edilmemiş ham `T×26` MSD segment özelliklerinden çıkarılan, fold eğitim kümesiyle normalize edilen değişken uzunluklu dizidir; padding mask ile yok sayılır.
- D girdileri: **audio + BPM + genre + musiXmatch lyrics**. Lyrics tam metin değil, MSD track ID ile eşleştirilmiş seyrek kelime sayımlarının train-only IDF ile üretilen 5.000 boyutlu TF-IDF gösterimidir. Eksik katalog MXM kaydı preflight hatasıdır; rapor boş doküman sayısını `0` kaydeder. Ayrı bir MXM coverage sayacı raporda yer almaz.
- `src/model_v1.py` içindeki ortak `NeuroTuneV1Encoder`, audio için `26→32→64→64` üç Conv1D katmanı, her katmandan sonra padding temizliği ve masked temporal mean pooling kullanır. Audio projeksiyonu 64-D; BPM dalı `1→8`, genre dalı `G→16`, lyrics dalı `5000→32` boyutludur. D füzyonu `120→128→64` üretir ve sonucu L2 normalize eder. C'nin lyrics içermeyen füzyonu `88→128→64`'tür; C ve D checkpoint'leri değiştirilebilir değildir.
- D encoder kolu önceden kodda vardı. Bu iş, A–D CV için MXM yükleme/coverage ve kaynak-hash doğrulamasını, train/validation-only lyrics özelliklerini, A/B/D eğitimini ve ayrı raporlamayı tamamladı. Eski C dosyaları yalnızca doğrulanmış read-only referanstır.

**Kod yolu:** `src/data/neurotune_v1_abcd_cv_training.py` (MXM ve fold özellikleri), `src/features_neurotune_v1.py` (train-only dönüşümler), `src/data/neurotune_v1_dataset.py` (ragged batch ve mask), `src/model_v1.py` (encoder), `src/triplets.py` (triplet/negatifler), `src/train_neurotune_v1.py` (eğitim/checkpoint seçimi) ve `scripts/run_neurotune_v1_full_modal_ablation.py` (izole koşu/rapor).

## Deney protokolü ve veri sınırı

- Üç artist-grouped fold (`0/1/2`), her fold'da üç seed (`684/685/686`), **50 epoch**, batch `64`, CPU'da Adam (`0.001`) ve cosine triplet-margin loss (`margin=0.2`): D için dokuz yeni eğitim koşusu. A ve B aynı bütçeyle eğitildi; C'nin dokuz tamamlanmış checkpoint'i yalnızca validation referansı oldu.
- Ses/BPM istatistikleri, genre sözlüğü ve lyrics IDF her fold'da **yalnızca train track'leriyle fit edilir**. Train ve validation hedef kenarları Last.fm train-source'dan gelir. Uniform negatif örnekleme aday havuzundaki bilinen pozitifleri dışlar. Official/legacy test-source pozitifler yalnızca bu negatif-hariç-tutma kümesinde bulunur; train/validation hedefi değildir. Bilinmeyen çiftler doğrulanmış negatif değildir.
- Checkpoint, validation **NDCG@10** en yüksek epoch'tan seçilir; eşitlikte validation loss ve sonra daha erken epoch kullanılır. Validation retrieval yalnızca validation galerisi ve kenarlarıyla yapılır. Runner mevcut test cohort'u için feature, embedding, relevance veya skor üretmez.
- A/B/C/D için her varyantın aynı modaliteleri kullanan eşit-ağırlıklı cosine baseline'ı vardır. Model ortalamaları **9 fold-seed**, cosine ortalamaları **3 fold** gözlemidir; bunları eşit sayıda bağımsız test gibi karşılaştırmıyoruz.

## Validation sonuçları

Tablo **ortalama ± popülasyon standart sapması** verir; metrik sırası Recall@10, HitRate@10, MRR@10, NDCG@10'dur. Ayrıntılı fold/seed değerleri ana JSON raporundadır.

| Varyant | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| A — yalnız audio | 0.048795 ± 0.005809 | 0.088177 ± 0.008653 | 0.034365 ± 0.005419 | 0.030221 ± 0.003879 |
| B — audio+BPM | 0.043948 ± 0.004498 | 0.078437 ± 0.005954 | 0.028857 ± 0.004429 | 0.026365 ± 0.003107 |
| C — audio+BPM+genre (read-only) | 0.118594 ± 0.010860 | 0.193172 ± 0.021044 | 0.069910 ± 0.006427 | 0.068638 ± 0.005853 |
| **D — audio+BPM+genre+lyrics** | **0.119240 ± 0.020447** | **0.194019 ± 0.027081** | **0.072913 ± 0.012158** | **0.071530 ± 0.012603** |

| Eşleştirilmiş cosine | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| A | 0.041842 ± 0.004772 | 0.075427 ± 0.007461 | 0.031908 ± 0.002566 | 0.027089 ± 0.003164 |
| B | 0.041812 ± 0.002146 | 0.080982 ± 0.000919 | 0.035022 ± 0.001153 | 0.028011 ± 0.000512 |
| C | 0.109255 ± 0.007844 | 0.190992 ± 0.012555 | 0.078063 ± 0.003241 | 0.068781 ± 0.004806 |
| **D** | **0.121607 ± 0.007892** | **0.204624 ± 0.011925** | **0.087984 ± 0.005050** | **0.077921 ± 0.006165** |

D'nin öğrenilmiş model ortalaması C'den dört metrikte de **küçük farkla yüksek**. Fakat aynı modaliteleri kullanan D cosine baseline, D modelini dört metrikte de geçiyor. Bu yüzden lyrics'in öğrenilmiş embedding'e güvenilir katkısı veya D'nin basit cosine'a üstünlüğü gösterilmiş değil. A/B'nin düşük olması da tek başına türün nedensel etkisini kanıtlamaz. D'nin seçilen epoch'ları seed sırası 684/685/686: fold 0 `44/2/37`, fold 1 `25/1/6`, fold 2 `20/42/2`; bu değişkenlik validation seçiminin duyarlılığını gösterir.

## Çıktılar, doğrulama ve yeniden çalıştırma

Yeni namespace `results/neurotune_v1_full_modal_ablation/` içinde `experiment.json`, `training_curves.csv` (**1.800 satır**), dokuz triplet schedule kopyası ve **27 yeni A/B/D checkpoint'i** bulunur (toplam 38 dosya). Rapor 36 model-validation ve 12 cosine-validation kaydı içerir; kaynak/kod/transform/schedule/checkpoint hash'lerini tutar. C'nin dokuz checkpoint'i, dokuz schedule'ı ve eğri CSV'si değiştirilmedi.

Bağımsız artifact kontrolünde **56 hash** eşleşti; **36/36** checkpoint'in variant/fold/seed/epoch, validation seçimi ve transform/schedule metadata'sı raporla eşleşti. Tam `unittest` suite'inde **423 test geçti**; `compileall` ve `pip check` başarılı. Uzun eğitimin işletim sistemi çıkış kodu host'tan geri alınamadı; `experiment.json` durumunun `complete` olması ve ayrı artifact/test doğrulamaları tamamlanma kanıtıdır. Test cohort'u değerlendirilmedi.

Koşu mevcut çıktı namespace'i varsa üzerine yazmayı reddeder. Mevcut raporu incelemek için yeniden eğitim gerekmiyor. Namespace boşken yeniden üretme komutu:

```bash
.venv/bin/python -m scripts.run_neurotune_v1_full_modal_ablation
```

## Bağımsız test kaynağına ilişkin son inceleme

Araştırmada [dim-sim](https://jongpillee.github.io/multi-dim-music-sim/) bulundu: [Zenodo kaydı](https://zenodo.org/records/3889149), MSD track ID'leriyle bağlı, insan tarafından karşılaştırılmış **879 temiz benzerlik üçlüsü**, CC BY-NC 4.0 lisansı. Bu etiketler Last.fm bağlantılarından farklıdır; ancak bunlar Top-10 pozitif kenarlar değil, üç saniyelik ses kliplerine ilişkin *üçlü tercihlerdir*. Yerel, salt-okunur ID sayımı mevcut 13.822 track'lik A–D katalogla **üç öğesi de eşleşen 0 temiz üçlü** buldu. Mevcut tüm MXM∩genre ID'lerinde bile yalnızca 9 temiz üçlünün tüm öğeleri var; üçünün de 3 saniyelik klibi ilk 30 saniye içinde kalanların sayısı 5. Dolayısıyla dim-sim'i mevcut D modelinin güvenilir yeni Top-10 testi gibi sunmuyor veya bu çalışmada kullanmıyoruz. Yeni etiketlerin toplanması ya da farklı, gerçekten eşleşen bir veri kaynağı ayrı karar gerektirir.

**Sonuç:** D uygulandı ve validation-only ablation tamamlandı. Yeni bağımsız test, bu teslimi tamamlamak için teknik önkoşul değil; yalnızca tarafsız genelleme/üstünlük iddiası için gerekli olan sonraki araştırma adımıdır.
