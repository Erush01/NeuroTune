# NeuroTune D4 — Son Test ve Doğrulama Kaydı

**Tarih:** 23 Kasım 2026  
**Kapsam:** İzole D4 multi-window audio doğrulaması; 3 fold × 3 seed, validation-only geliştirme koşusu.

## Test ve preflight sonuçları

| Kontrol | Komut / kapsam | Sonuç |
|---|---|---|
| Tam birim test paketi | `.venv/bin/python -m unittest discover -s tests -q` | **546 test geçti**, çıkış kodu 0 |
| Python sözdizimi/derleme | `.venv/bin/python -m compileall -q src scripts tests` | Başarılı, çıkış kodu 0 |
| Bağımlılık tutarlılığı | `.venv/bin/python -m pip check` | Sorun bulunmadı, çıkış kodu 0 |
| CLI yardım kontrolleri | D4 cache preprocessor ve D4 runner `--help` | İkisi de başarılı, çıkış kodu 0 |
| D4 odaklı testler | 10 odaklı test paketi | **163 test geçti** |
| Gerçek model/trainer sentetik smoke testi | D4 multi-window batch ile model, eğitim ve validation | Başarılı; test verisi kullanılmadı |
| Üç fold train-only feature preflight'i | D4 cache ve D0 fold manifest'leri | Her fold'da yalnız train+validation track'leri üretildi; test ID'leri feature builder'a verilmedi |

Tam test paketi, D4 kod entegrasyonundan sonra ve üretim cache/eğitim koşularından önce çalıştırıldı. Sonraki değişiklikler yalnızca README ve Markdown raporlarıydı; kaynak kod değiştirilmedi.

## Cache üretimi

Çalıştırılan komut:

```bash
/usr/bin/time -v -o /tmp/opencode/neurotune_d4_cache_timing.txt \
  .venv/bin/python -m scripts.preprocess_neurotune_v1_d4_audio_windows \
  > /tmp/opencode/neurotune_d4_cache_run.log 2>&1
```

- Çıkış kodu: **0**; süre: **5:11.40**; tepe RSS: **1,096,616 KiB**.
- Cache: `data/msd_hf_abcd_pilot/abcd_audio_start_mid_end_30s_26ch.h5`.
- **13,822 track**, **4,003,002** float32 özellik satırı, **315.18 MiB**; reddedilen track: **0**.
- Başlangıç/orta/son pencerelerde geçerli track sayıları: **13,822 / 13,813 / 13,813**. **13,804** track'te üç, **18** track'te iki pencere geçerliydi; geçerli penceresi olmayan track yoktu.
- D4 track ID'leri D0 cache ID'leriyle eşleşti. Pencere başlangıçları, offset'ler, validity, timestamp ve metadata tutarlılık kontrolleri geçti.

## Sabit pilot

Fold 0 / seed 684 için sabit pilot başarıyla tamamlandı. Pilot ayrı namespace'e tek checkpoint, 50 sonlu epoch kaydı ve D0'dan byte-byte aynı bir schedule kopyası yazdı. Seçilen epoch 10, validation NDCG@10 **0.092061** oldu. Çalışma süresi **2:21:03**, tepe RSS **2,364,004 KiB** idi. Pilot yalnızca kaynak/uygulama kontrolü olarak kullanıldı; ayar değişikliği için kullanılmadı.

## Tam D4 koşusu

Çalıştırılan komut:

```bash
/usr/bin/time -v -o /tmp/opencode/neurotune_d4_full_timing.txt \
  .venv/bin/python -m scripts.run_neurotune_v1_d4_multi_window_audio \
  > /tmp/opencode/neurotune_d4_full_run.log 2>&1
```

- Çıkış kodu: **0**; süre: **15:58:25**; tepe RSS: **2,959,092 KiB**.
- Kapsam: fold 0/1/2 × seed 684/685/686; **9/9** model kaydı ve checkpoint, **9/9** D0 schedule kopyası, **450** training-curve satırı ve **3** D4-matched cosine fold kaydı.
- Metrikler sonlu; D4 checkpoint, schedule, transform ve training-curve hash'leri kayıtlı manifest'lerle eşleşti. Runner'ın korunan girdi/referans/kod hash kontrolü geçti; D0–D3 sonuç dosyalarının kayıtlı çıktı hash'leri de doğrulandı.
- `results/neurotune_v1_d4_multi_window_audio/experiment.json` durumu `complete`; `independent_test_artifacts_created` değeri `false`.

### Validation ortalamaları

| Yöntem | Gözlem sayısı | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| D0 modeli | 9 fold-seed | 0.119240 | 0.194019 | 0.072913 | 0.071530 |
| D4 modeli | 9 fold-seed | 0.127762 | 0.205434 | 0.076902 | 0.075412 |
| D0-matched cosine | 3 fold | 0.121607 | 0.204624 | 0.087984 | 0.077921 |
| D4-matched cosine | 3 fold | 0.128981 | 0.219120 | 0.089028 | 0.080621 |

D4−D0 fold-ortalaması NDCG@10 farkları fold 0/1/2 için **+0.004770 / +0.000460 / +0.006415** oldu. Dokuz paired seed karşılaştırmasının sekizi D4 lehineydi; fold 1 / seed 686 farkı **−0.001891** idi. D4 model ortalaması D0'dan yüksek olsa da D4-matched cosine ortalaması dört metriğin tümünde daha yüksekti; model NDCG@10 değeri fold 1 ve 2'de D4 cosine'ın altındaydı.

## D0–D4 artefakt bütünlüğü

Tam koşu sonrasında her deneyin kendi `experiment.json` çıktı manifest'iyle dosya hash'leri karşılaştırıldı:

| Deney | Doğrulanan çıktı/artefakt hash girdisi | Sonuç |
|---|---:|---|
| D0 | 54 | Eşleşti |
| D1 | 40 | Eşleşti |
| D2 | 21 | Eşleşti |
| D3 | 23 | Eşleşti |
| D4 | 9 checkpoint + 9 schedule + 3 transform + training-curve hash'i | Eşleşti |

D4 çıktıları `results/neurotune_v1_d4_multi_window_audio/` altındadır; cache, D0'ın ilk-pencere cache'ini değiştirmeyen ayrı bir dosyadır. D0–D3 sonuç namespace'lerine yazılmadı.

## Yorum ve sınırlar

Bu değerler validation-only geliştirme sonuçlarıdır; validation checkpoint seçimi için de kullanıldı ve fold'lar bağımsız örnekler değildir. Dokuz model koşusu ile üç fold-level cosine gözlemi aynı sayıda gözlem değildir. Bu nedenle sonuçlar model üstünlüğü veya genelleme kanıtı sayılmaz. Bağımsız test özellikleri, embedding'leri, relevance etiketleri veya metrikleri oluşturulmadı. D0'dan devralınan official/legacy test-source pozitifleri yalnızca train-local negatif adaylarını elemek için kullanılabildi; hedef veya değerlendirme verisi olmadı.

## İlgili kayıtlar

- Deney: `results/neurotune_v1_d4_multi_window_audio/experiment.json`
- Training curve: `results/neurotune_v1_d4_multi_window_audio/training_curves.csv`
- Cache metadata ve rejection sidecar'ları: `data/msd_hf_abcd_pilot/`
- Uygulama planı: `docs/plans/2026-11-22-neurotune-d4-multi-window-audio.md`
- Sonuç ve çalıştırma dokümantasyonu: `README.md`, `NeuroTune_v1_Performans_Iyilestirme_Deney_Plani.md`
