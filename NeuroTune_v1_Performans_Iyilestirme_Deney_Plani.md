# NeuroTune v1 - Performans İyileştirme ve Kontrollü Deney Planı

**Tarih:** 18–23 Kasım 2026  
**Durum:** D1–D4 validation-only geliştirme koşuları tamamlandı; bağımsız test yapılmadı. D4 tam koşusu 15:58:25 sürdü. D4 modelinin ortalama NDCG@10 değeri D0'dan yüksekti (0.075412 vs. 0.071530), ancak D4-matched cosine ortalamasının altındaydı (0.080621); fold 1 ve 2'de de model cosine'dan düşüktü. Bu gözlemler model üstünlüğü veya genelleme kanıtı değildir.
**Kapsam:** D1 semi-hard negative mining, D2 modality dropout, D3 birleşik eğitim ve D0'a karşı izole D4 çok pencereli audio karşılaştırması  
**Hedef:** Audio + BPM + genre + lyrics girdili NeuroTune D modelinin embedding kalitesini iyileştirmek ve iyileşmeyi baseline'a karşı adil bicimde ölçmek.

> **Temel karar:** Yeni ve daha büyük bir neural network kurmadan önce mevcut 1D-CNN + multimodal fusion mimarisinin *eğitim stratejisini* iyileştirecegiz. Başarıyi peşinen varsaymiyoruz. Tüm yeni model seçimi geliştirme verisiyle yapılacak; daha önce incelenmiş test kohortu tekrar optimizasyon için kullanılmayacak.

---

## 1. Mevcut referans: ne biliyoruz?

Kaynaklar:

- `NeuroTune_D_Modeli_Uygulama_ve_Validation_Sonuclari_2026-11-17.md`
- `NeuroTune_D_Modalite_Duyarliligi_2026-11-17.md`
- `NeuroTune_durum_raporu_2026-11-16.md`

### 1.1 Model ve veri

- MSD A-D pilot katalog: **13.822 track**.
- Audio: ilk 30 saniyeye ait değişken uzunluklu **T x 26** segment özellikleri; train-only normalizasyon, padded segmentler maskeleniyor.
- Model: `26 -> 32 -> 64 -> 64` Conv1D + masked temporal mean pooling, **64D audio**; BPM `1 -> 8`, genre `G -> 16`, lyrics TF-IDF `5000 -> 32`.
- D fusion: `120 -> 128 -> 64`, son katmanda L2 normalization.
- Eğitim: cosine triplet-margin loss (`margin=0.2`), Adam (`lr=0.001`), batch 64, en fazla 50 epoch.
- Deneyler: 3 artışt-grouped fold (`0/1/2`) x 3 seed (`684/685/686`). Checkpoint seçimi validation NDCG@10 ile.
- Lyrics: tam metin değil; musiXmatch kelime sayımlarından train-only IDF ile üretilmis TF-IDF.

### 1.2 Baseline performansi - validation

| Metrik | D neural | D matched cosine | Neural - cosine |
|---|---:|---:|---:|
| Recall@10 | 0.119240 | 0.121607 | -0.002367 |
| HitRate@10 | 0.194019 | 0.204624 | -0.010605 |
| MRR@10 | 0.072913 | 0.087984 | -0.015071 |
| NDCG@10 | 0.071530 | 0.077921 | **-0.006391** |

Degerler mevcut rapordan alinmis validation ortalamalaridir. Neural satirlar 9 fold-seed kosusuna, cosine satirlar 3 fold'a dayanir; bunlar 9 vs. 3 bağımsız test gozlemi sayilmamalidir. Ayni fold üzerinde paired karşılaştırma yapılmalidir.

### 1.3 Modalite duyarliligi - validation-only tanısal test

| Sifirlanan dal | NDCG@10 | Maskesiz D'ye göre NDCG kaybı | Top-10 ortusme |
|---|---:|---:|---:|
| Yok | 0.071530 | 0 | 1.000000 |
| Audio | 0.063567 | 0.007963 | 0.586110 |
| BPM | 0.070262 | 0.001268 | 0.861922 |
| Genre | 0.022969 | 0.048561 | 0.265619 |
| Lyrics | 0.052619 | 0.018911 | 0.124028 |

**Yorum:** Modelin genre ve lyrics sıfırlanmasına duyarliligi yüksek; audio dalinin da ölçülebilir etkisi var. Fakat sıfır vektorleri dağılim disi girdiler olabilecegi için bu bulgular *nedensel feature importance* veya gerçek muziksel önem sıralamasi degildir.

### 1.4 Cozulmesini istedigimiz problemler

1. Egitimden öğrenilen D embedding'leri matched cosine baseline'i tutarlı bicimde gecmiyor.
2. Uniform random negative sampling muhtemelen kolay triplet uretip ayrım gücünü sınırliyor olabilir; bu bir **hipotez**.
3. Model genre/lyrics girdilerine yüksek duyarlilik gösteriyor; audio temsilinin kullanımı artırılabilir mi?
4. Audio encoder yalnızca ilk 30 saniyeyi goruyor; şarkının diger bolumlerinde ayirt edici bilgi bulunabilir.
5. Last.fm pozitif grafiği eksık: bağlantı yoklugu **doğrulanmış negatif** demek değil.

---

## 2. Öncelikli yol haritasi

| ID | Deney | Neyi değiştiriyoruz? | Başlangıç parametresi | Öncelik |
|---|---|---|---|---|
| D0 | Mevcut D | Hicbir sey; sabit referans | Uniform random; dropout yok; ilk 30 sn | Referans |
| **D1** | Semi-hard negative mining | Yalnız negative sampler | %75 random / %25 semi-hard | **P0** |
| **D2** | Modality dropout | Yalnız egitimde yardimci modaliteler | Genre %25; lyrics %20; BPM %10; audio %0 | **P1** |
| **D3** | D1 + D2 | Iki iyileştirmenin birlikte etkisi | D1 ve D2'nin önceden tanimli ayarlari | **P1** |
| D4 | Multi-window audio | Sarkinin farklı zaman bolumleri | Başlangıç + orta + son, 30 sn pencere | P2 / tamamlandı |

**Sirali uygulama:** D1 -> D2 -> D3 tamamlandı. D4 daha sonra, D3 değişikliklerini birleştirmeden ve yalnızca audio pencere temsilini değiştirerek D0'a karşı bağımsız bir karşılaştırma olarak çalıştırıldı.

D1, D2, D3 ve D4 birbirinin ardindan aynı modeli fine-tune eden asamalar değil; her biri **D0 ile aynı başlangıç protokolunden bağımsız eğitim varyantlari** olmalidir. D4, D1/D2/D3 değişikliklerini birleştirmez. Bu, değişikliklerin etkisini ayrıştırmak için gerekli.

---

## 3. Deney D1 - Semi-hard negative mining

### 3.1 Amac ve tanım

Gecis, mevcut uniform negative sampling'in bir kismini daha bilgilendirici negatiflerle değiştirmek. Anchor `a`, bilinen pozitif `p`, negatif adayi `n`, cosine distance `d`, margin `m=0.2` olmak uzere klasık semi-hard kosulu:

```text
d(a, p) < d(a, n) < d(a, p) + m
```

Bu banddaki adaylar kolay negatiflere göre daha fazla ayrım sinyali üretebilir. Fakat **Last.fm'de bağlantısı bulunmayan adaylar kesin negatif degildir**; false-negative riski kontrol altinda tutulmalidir.

### 3.2 Önerilen ilk protokol

- Negative oranları: **%75 uniform random + %25 semi-hard**. Bu sadece başlangıç denemesidir, optimize edilmis değer degildir.
- Ilk **3 epoch** uniform random warm-up; sonra belli araliklarla mevcut encoder ile train track embedding'lerini güncelle.
- Mining araligi için ilk deneme: her **5 epoch**; bağımsız kontrol / audit kaydet.
- Mining yaparken encoder `eval`/`no_grad`; model eğitim durumu ve RNG deterministik sekilde geri yuklenmeli.
- Yalnız **ilgili fold'un train artışt/track kataloğu** üzerinden candidate embedding hesapla.
- Bilinen pozitifleri D0'daki gibi ilgili train aday havuzundan kesin dışla. Kullanıcının onayladığı D1 karşılaştırmasında, train havuzuna düşen official/legacy test-source pozitifleri yalnızca negatif adayı elemek için kullanılabilir; bunlar hedef, özellik veya metrik değildir. Test track özellikleri ve embedding'leri mining'e girmez.
- Aynı track, duplicate track ve veri temizlik kurallarina takilan adaylari disla.
- Yanlis-negatif riski tasiyan adaylar için muhafazakar politika kullan: gerektiğinde semi-hard adayi bulamayinca **uniform random'a fallback**.
- Mevcut `triplet schedule` yapisini koru: fold, seed, epoch, anchor, positive, negative, sampling type ve mining version audit edilebilir olmali.
- Random negatiflerin oranı hicbir zaman sıfıra düşmemeli. Hardest-negative seçimini ilk deneyde kullanma.

> **Kritik not:** Kullanıcı, bu D1 validation-development karşılaştırmasında D0'ın mevcut known-positive exclusion politikasının korunmasını onayladı. Buna göre train havuzuna düşen official/legacy test-source pozitifleri yalnızca negatif adayları dışlamak için kullanılabilir; hedef olarak kullanılmaz ve bu istisna raporda açıkça belirtilir. Bu karar yeni ve gerçekten bağımsız bir test kohortuna uygulanmaz: gelecekteki bağımsız test etiketleri eğitim, mining veya checkpoint kararlarına hiçbir amaçla girmemelidir.

### 3.3 Uygulama görevleri

1. `src/triplets.py` içinde veya yeni modulde arayuzu tanimli, seed'li, test edilebilir `uniform` ve `mixed_semi_hard` sampler'lari olustur.
2. `src/train_neurotune_v1.py` training loop'una train-only embedding yenileme noktasi ekle. Mevcut default davranis **değişmemeli**.
3. Bir run konfigürasyonunda `negative_strategy`, `semi_hard_fraction`, `warmup_epochs`, `refresh_interval`, `margin`, `rng_seed` alanlarini kaydet.
4. Semi-hard bulunamayan anchor oranini, fallback sayisini ve gerçeklesmis random/semi-hard oranini raporla.
5. Yeni runner/namespace kullan; mevcut D0 checkpoint, schedule, manifest ve raporlari overwrite etme.

### 3.4 Eğitim loglarinda ek alanlar

- Toplam triplet sayısı; positive/negative sayısı ve anchor başına dağılim.
- **Aktif triplet oranı:** `loss > 0` olan orneklerin yuzdesi.
- `d(a,p)` ve `d(a,n)` istatistikleri, margin violation rate.
- Mining candidate havuzu büyüklugu, kabul/red/fallback sayilari.
- Mining'le seçilen negatiflerin benzerlik/genre dağılimi (tanısal; genre etiketi yoksa logla).
- Epoch-bazli train loss, validation NDCG@10, diger retrieval metrikleri.
- Fold/seed bazli runtime, training schedule hash ve checkpoint hash.

### 3.5 D1 tamamlanma kriterleri

- Sampler deterministik, test edilmis ve train-fold disina erismiyor.
- Bilinen train pozitiflerini hicbir zaman negatif olarak secmiyor.
- Normal şartlarda fallback çalışiyor; bos semi-hard havuzunda eğitim çokmuyor.
- D0 ile aynı data/split/encoder/optimizer/epoch butcesinde kosuluyor.
- Validation sonucunda D0 ve D matched cosine'a **paired** farklar raporlaniyor; kazanacagi varsayilmiyor.

---

## 4. Deney D2 - Modality dropout

### 4.1 Amac

D modelinin genre/lyrics bilgisini kullanirken audio temsilinden de faydalanmasini tesvik etmek. Ayni mimari, aynı 30 sn audio ve mevcut **uniform negative sampling** korunacak. Tek değişıklik: **yalnız training sırasında** yardimci modalitelerin kontrollü kapatilmasi.

| Dal | Ilk deneme dropout olasılığı |
|---|---:|
| Audio | **0.00** |
| BPM | **0.10** |
| Genre | **0.25** |
| Lyrics | **0.20** |

Bu oranlar başlangıç hipotezidir, literaturden doğrulanmış optimum değerler degildir.

### 4.2 Doğru uygulama

- Dropout **yalnızca train forward** için etkin olmali; validation ve inference'da tüm modaliteler açık.
- BPM/genre/lyrics maskesi her örneğin **branch aktivasyonundan sonra, fusion'dan hemen önceki embedding'ine** uygulanir; dal kapatildiginda embedding tamamen sifirlanir.
- Ters olasılık ölçeklemesi uygulanmaz; bu seçimi config ve provenance'da kaydet.
- Audio ve audio padding mask dokunulmaz; tüm modaliteleri aynı anda kapatan durumdan kaçın.
- Dropout rastgeleliği bağımsız CPU `torch.Generator` ile üretilir: her epoch seed'i `seed * 1000003 + fold * 1009 + epoch_index_zero_based`; batch/rol/modality tüketim sırası deterministiktir ve triplet-shuffle RNG'sini etkilemez.
- Kaybolan modalite durumu inference sırasında beklenen bir girdi degildir; 17 Kasım'daki `zero masking` tanısal analizle karistirma.
- Genre ve lyrics'in model tarafindan tamamen goz ardi edildigini varsayma: **amac**, genel retrieval sonucunu iyileştirmek.

### 4.3 Tamamlanma kriterleri

- Maskelerin sadece train'de ve önceden belirlenen olasıliklarla uygulandigini test et.
- D0 ile train/validation protokolunu eşleştir.
- Validation NDCG@10, Recall@10, MRR@10, HitRate@10 ve eğitim kararliligini karşılaştır.
- Mevcut modalite duyarlilik analizi ile uyumlu yeni validation-only tanısal rapor üret; genelleme veya nedensel özellik önemi iddiasi kurma.

### 4.4 Uygulama ve sonuç

D2, `config/neurotune_v1_d2_modality_dropout.json` ve `scripts/run_neurotune_v1_modality_dropout.py` ile bağımsız D varyantı olarak tamamlandı. D0'ın dokuz uniform schedule dosyası byte-for-byte yeniden kullanıldı; D0 checkpoint/schedule referansları ve hash'leri doğrulandı. Sonuçlar `results/neurotune_v1_d2_modality_dropout/` altındadır: dokuz checkpoint, dokuz schedule kopyası, 450 epoch eğrisi satırı ve 450 modality-dropout tanı satırı. Bağımsız test verisi kullanılmadı.

Dokuz fold-seed gözlemi üzerindeki ortalama validation metrikleri:

| Model | Gözlem | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| D0 model | 9 | 0.1192 | 0.1940 | 0.0729 | 0.0715 |
| D2 modality dropout | 9 | 0.1410 | 0.2263 | 0.0820 | 0.0822 |
| D-matched cosine | 3 fold | 0.1216 | 0.2046 | 0.0880 | 0.0779 |

Üç seed'in fold ortalamaları için NDCG@10 (D2 / D0 / cosine) sırasıyla fold 0'da `0.0969 / 0.0890 / 0.0854`, fold 1'de `0.0709 / 0.0614 / 0.0703`, fold 2'de `0.0788 / 0.0643 / 0.0780` oldu. D2'nin gözlenen NDCG@10 ve Recall@10 fold ortalamaları bu sabit koşuda D0 ve fold-level cosine değerlerinin üzerindeydi; MRR@10 ise cosine'ın fold 1 ve 2 değerlerinin altında kaldı. Bunlar validation geliştirme bulgularıdır; validation checkpoint seçimi için de kullanıldığı, fold'lar bağımsız örnekler olmadığı ve yeni bağımsız test yapılmadığı için model üstünlüğü/genelleme kanıtı sayılmaz.

Gerçekleşen maskeleme oranları BPM için `0.099991`, genre için `0.250032`, lyrics için `0.199956` oldu (her dalda 131,202,900 track-role maskesi). Tam koşu 4:51:55.5 sürdü. Tüm metrik, fold/seed, schedule/checkpoint/output ve kod hash'leri `results/neurotune_v1_d2_modality_dropout/experiment.json`, `training_curves.csv` ve `modality_dropout_diagnostics.csv` içinde tutulur. D3 sonuçları 5. bölümde verilmiştir.

---

## 5. Deney D3 - Semi-hard mining + modality dropout

- D1 ve D2'nin **önceden belirtilen** ayarlarini birlikte uygula.
- Yeni fold/seed bağımsız D3 kosulari üret; D1 checkpoint'inden egitime devam ederek D3'u D1 ile karistirma.
- D0 vs D1, D0 vs D2, D0 vs D3 ve D3 vs D1/D2 farklarini aynı splitler üzerinde raporla.
- D1 ve D2 tek başına faydali olmadan da kombinasyon etkisi olabilir; ancak D3'u yalnızca validasyon sonuclarina bakarak tekrar tekrar yeniden ayarlama.
- Eger ikili strateji daha kotu cikarsa D0 veya tek değişıklikli modeli koru; 'daha kompleks = daha iyi' kabul etme.

### 5.1 Uygulama ve sonuç

D3, `config/neurotune_v1_d3_combined.json` ve `scripts/run_neurotune_v1_d3_combined.py` ile bağımsız D varyantı olarak tamamlandı. Her fold/seed için model baştan başlatıldı ve trainer'a D0/D1/D2 ile eşleşen seed (`684/685/686`) verildi; D0, D1 veya D2 checkpoint'lerinden devam edilmedi. D0 uniform schedule'ları ilk üç epoch'ta kullanıldı; sonraki epoch'larda D1 semi-hard sağlayıcısı ve D2 training-only branch dropout birlikte uygulandı. D3, aynı fold/seed üzerindeki D0/D1/D2 değerleri ve fold-level D-cosine ile karşılaştırıldı.

Dokuz fold-seed gözlemi üzerindeki ortalama validation metrikleri:

| Model | Gözlem | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| D0 model | 9 | 0.1192 | 0.1940 | 0.0729 | 0.0715 |
| D1 semi-hard | 9 | 0.1230 | 0.1948 | 0.0712 | 0.0722 |
| D2 modality dropout | 9 | 0.1410 | 0.2263 | 0.0820 | 0.0822 |
| D3 combined | 9 | 0.1403 | 0.2227 | 0.0812 | 0.0822 |
| D-matched cosine | 3 fold | 0.1216 | 0.2046 | 0.0880 | 0.0779 |

Üç seed'in fold ortalamaları için NDCG@10 (D3 / D0 / D1 / D2 / cosine) fold 0'da `0.0983 / 0.0890 / 0.0919 / 0.0969 / 0.0854`, fold 1'de `0.0709 / 0.0614 / 0.0595 / 0.0709 / 0.0703`, fold 2'de `0.0774 / 0.0643 / 0.0652 / 0.0788 / 0.0780` oldu. D3 ortalama NDCG@10 değeri `0.082233`, D2 değeri `0.082199` idi; fark çok küçüktü. D3'ün Recall@10, HitRate@10 ve MRR@10 ortalamaları D2'nin altında kaldı. D3-D2 fold NDCG farkları `+0.001437 / 0 / -0.001337` oldu; kombinasyon tek değişiklikli D2'ye karşı tutarlı kazanım göstermedi.

Dokuz D3 checkpoint'inin altısı epoch 1–3'te, semi-hard mining'in başladığı 4. epoch'tan önce seçildi. Bunlardan beşinin ağırlıkları D2 karşılıklarıyla aynıydı: fold 1'de seçilen epoch'lar 2, 1 ve 2; fold 0/seed 686 ile fold 2/seed 686'da seçilen epoch 3. Fold 2/seed 685'te D3 epoch 1'i, D2 ise epoch 29'u seçtiği için checkpoint'ler farklıydı. D3 ve D2 warm-up sırasında aynı seed, dropout stream'i ve uniform schedule'ı kullandı. Gerçekleşen semi-hard payı 43,734,300 triplet içinde `%23.5001` oldu (üç uniform warm-up epoch'u nedeniyle); beş uniform fallback görüldü. Gerçekleşen dropout oranları BPM için `0.099991`, genre için `0.250032`, lyrics için `0.199956` idi (her dalda 131,202,900 track-role maskesi). Tam koşu 8:03:57 sürdü; epoch süresi medyanı 46.35 sn, aralığı 37.46–154.25 sn oldu.

Çıktılar `results/neurotune_v1_d3_combined/` altındadır: dokuz checkpoint, dokuz epoch-schedule arşivi, 450 satırlık eğitim eğrisi, mining ve dropout tanı dosyaları, paired metrikler ve `experiment.json` içindeki provenance/hash'ler. Bağımsız test kullanılmadı. Bulgular validation-only geliştirme sonuçlarıdır; fold'lar bağımsız örnekler değildir ve model üstünlüğü/genelleme kanıtı sayılmaz.

---

## 6. Deney D4 - Multi-window audio (validation-only development)

D4, D3 sonrasında onaylandı ve yalnızca audio penceresi temsilinin etkisini ölçmek için D0'a karşı bağımsız çalıştırıldı. D1 semi-hard mining veya D2/D3 modality dropout ile birleştirilmedi; her fold/seed koşusu yeni model ağırlıklarıyla başladı ve D0 model ağırlıkları yüklenmedi.

### 6.1 Kilitlenen yöntem

- Her parça için timestamp tabanlı üç 30 saniyelik pencere kullanıldı: başlangıç `0`, orta `max(0, (duration - 30) / 2)`, son `max(0, duration - 30)`. Pencereler `[start, start + 30)` aralığıdır; doğal overlap korunur.
- 30 saniyeden kısa parçalar mevcut açılış sekansını üç görünümde de kullanır. Boş pencereler maskelenir; yalnızca geçerli pencere temsilleri ortalanır. Üç pencerenin tümü geçersiz olan bir track fail-fast ile reddedilir.
- D0'ın CNN/projection mimarisi değiştirilmedi; yeni D4 encoder ağırlıkları üç pencere arasında paylaşılır ve D0 checkpoint ağırlıkları yüklenmez. Ham 64D projection vektörleri fusion'dan önce ortalanır; D0'ın fusion yapısı ve yalnızca son birleşik embedding üzerindeki L2 normalizasyonu korunur.
- Audio kanal ortalama/std değerleri fold'un yalnızca train track'lerinden fit edilir. Overlap nedeniyle tekrar görülen `(track_id, source_segment_index)` satırları istatistik için bir kez, tekrarların eleman bazlı ortalamasıyla sayılır; model girdilerindeki özgün pencere satırları değiştirilmez.
- Cohort (13.822 track), artist-grouped fold'lar, etiketler, candidate pool'lar, D0 uniform schedule'ları, seed'ler (684/685/686), Adam (`lr=0.001`), margin (`0.2`), batch (64), 50 epoch ve validation NDCG@10 checkpoint seçimi aynı tutuldu. Eşleşmiş D4 cosine baseline da aynı üç pencere temsilinden yeniden hesaplandı.

### 6.2 Cache, çalıştırma ve kaynak maliyeti

Yeni cache `data/msd_hf_abcd_pilot/abcd_audio_start_mid_end_30s_26ch.h5` ve ayrı metadata/rejection sidecar'larında tutulur; D0'ın ilk-pencere cache'i değiştirilmedi. Cache 13.822 track ve 4.003.002 float32 özellik satırı içerir (315,18 MiB); sıfır track reddedildi. Başlangıç/orta/son geçerli pencere sayıları sırasıyla 13.822/13.813/13.813; 13.804 track'te üç, 18 track'te iki pencere geçerlidir. Pencere başlangıçları, offset'ler, validity işaretleri, timestamp'ler, cohort ID'leri ve source/code hash'leri preflight sırasında doğrulandı.

Cache üretimi 5:11.40 sürdü ve tepe RSS 1.096.616 KiB oldu. Fold-local train-only transform kontrollerinde her fold'un D4 feature ID'leri tam olarak train+validation ID'leriydi; test ID'leri feature builder'a verilmedi. Deduplicated audio fit-row sayıları fold 0/1/2 için 3.352.109 / 3.369.552 / 3.364.986 oldu; değerler cache'teki benzersiz train `(track_id, source_segment_index)` sayılarıyla eşleşti. Non-audio transform'lar D0 ile aynı kaldı.

Sabit pilot fold 0/seed 684 ile çalıştırıldı; ayrı namespace'e tek checkpoint ve byte-identical D0 schedule yazıldı. Ardından tam 3 fold × 3 seed koşusu bir kez çalıştırıldı; çalışma 15:58:25 sürdü ve tepe RSS 2.959.092 KiB oldu. Tam sonuçlar `results/neurotune_v1_d4_multi_window_audio/experiment.json` ve `training_curves.csv` içindedir; dokuz checkpoint, dokuz D0 schedule kopyası ve 450 epoch satırı doğrulandı. D0 referansları, kaynaklar, kod, dönüşümler ve tüm çıktı hash'leri kontrol edildi; D0–D3 sonuç artefaktları kendi kayıtlı hash'leriyle eşleşti.

### 6.3 Validation sonuçları

| Yöntem | Gözlem | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| D0 model | 9 fold-seed | 0.1192 | 0.1940 | 0.0729 | 0.0715 |
| D4 model | 9 fold-seed | 0.1278 | 0.2054 | 0.0769 | 0.0754 |
| D0 eşleşmiş cosine | 3 fold | 0.1216 | 0.2046 | 0.0880 | 0.0779 |
| D4 eşleşmiş cosine | 3 fold | 0.1290 | 0.2191 | 0.0890 | 0.0806 |

Üç seed'in fold ortalaması NDCG@10 değerleri:

| Fold | D4 model | D0 model | D4 cosine | D0 cosine | D4 − D0 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0937 | 0.0890 | 0.0892 | 0.0854 | +0.0048 |
| 1 | 0.0618 | 0.0614 | 0.0725 | 0.0703 | +0.0005 |
| 2 | 0.0707 | 0.0643 | 0.0802 | 0.0780 | +0.0064 |

D4 modelinin ortalama dört metriği D0'dan yüksekti; paired NDCG@10 farkı 9 seed karşılaştırmasının 8'inde pozitifti. Fold ortalamaları da D0'dan yüksekti; en küçük artış fold 1'deydi. Buna karşın D4 modelinin genel ortalamaları D4 cosine ortalamalarının altındaydı ve model fold 1/2'de matched cosine'dan düşük NDCG@10 aldı. Model ortalamaları dokuz, cosine ortalamaları üç gözleme dayanır; bu değerler inferential karşılaştırma değildir. Seçilen epoch'lar 1–25 aralığındaydı (medyan 10).

### 6.4 Yorum ve sınırlar

D4'ün gözlenen validation farkı audio pencere değişikliğinin bu sabit geliştirme koşusundaki sonucu olarak raporlanmalıdır; model üstünlüğü, nedensel etki veya genelleme olarak yorumlanmamalıdır. Validation checkpoint seçimi için kullanıldı ve fold'lar bağımsız örnekler değildir. Bağımsız test özellikleri, embedding'leri, relevance etiketleri veya metrikleri üretilmedi. D0'dan devralınan known-positive exclusion politikası korundu: official/legacy test-source pozitifleri yalnızca train-local negatif adaylarını dışlayabilir; hedef veya değerlendirme metriği değildir. Gelecekteki bağımsız test cohort'u etiketleri eğitim veya negatif mining'e hiçbir amaçla girmemelidir.

---

## 7. Deney tasarimi ve veri sizintisi kuralları

### 7.1 Degismez prensipler

1. **Mevcut A-D test kohortu yeni hiperparametre ayari için tekrar açılmayacak.** D1–D4 geliştirme odaklı deneylerdir.
2. Yeni tamamen bağımsız test kohortu için hem **artışt/track overlap** hem de **relevance/etiket kaynak sizintisi** denetlenmeli.
3. Yeni bağımsız bir test kohortunun etiketleri train veya negative mining'e hiçbir amaçla sokulmamalı. D1–D4 geliştirme karşılaştırmaları, D0'dan devralınan eski official/legacy test-source pozitiflerini yalnız train-local negatif adaylarını dışlamak için kullanabilir; bu sınır ayrıca raporlanır.
4. Fold-bazli tüm train-only feature transformlar (audio/BPM normalizasyonu, genre sozlugu, lyrics IDF) yeniden yalnız ilgili train tracklerinden fit edilmeli.
5. Validation NDCG@10 checkpoint seçimi için zaten kullanıldigindan, aynı validation skorlarini **bağımsız test** gibi raporlama.
6. Başlangıç D0 protokolundeki tüm input hashes/split IDs kaydedilip yeni kosularda karşılastirilmali.
7. Eski `results/` namespace'leri **read-only** kalmali; yeni cikti dizini var ise fail-fast.
8. Bilinmeyen Last.fm ciftleri gerçek negatif kabul edilmeyecek; negative mining etkisini bu sınırla yorumla.
9. Test/validation track embedding'leri negative mining aday havuzuna dahil edilmeyecek.
10. Bir bağımsız test mevcut degilse sonuçları **validation-based development result** diye raporla.

### 7.2 Kademeli kosu planı

**Hızlı doğrulama:**

- Önce sentetik/mini fixture üzerinde sampler, dropout ve leakage kontrolleri.
- Sonra **fold 0, seed 684/685/686**, en fazla 50 epoch, tüm D varyantlari için aynı bütçe.
- Başlangıç ayarlari **önceden sabit**: D1 %25 semi-hard, 3 warm-up epoch, 5 epoch mining refresh; D2 tabloda yazan dropout.
- Degisken eğitim sürelerini ve checkpoint seçimi farklarini kaydet.

**Tam geliştirme degerlendirmesi:**

- Önce D1/D2/D3 stratejilerinin değerlendirilecegi sabit protokol ve karar kurallarini belgeleyip kilitle.
- Ardindan uygun adaylari **fold 0/1/2 ve seed 684/685/686** için eğit.
- Yeni modelleri D0 ve matched cosine ile aynı validation galeri/relevance kurallarinda karşılaştır.
- Fold 0 görüldugu için tüm 3-fold ortalamalar da tamamen bağımsız delil olmayacak; sonuç geliştirme bulgusu olarak kalacak.

### 7.3 Metrikler ve karşılaştırma

**Birincil:** NDCG@10.  
**Ikincil:** Recall@10, HitRate@10, MRR@10.  
**Tani:** aktif triplet %, positive/negative distance, margin violation, Top-10 overlap, embedding drift, genre ici retrieval, modality mask duyarliligi, epoch seçimi ve loss egrileri.

Rapor tablosu asgari su alanlari icermeli:

| Model | Fold | Seed | Secilen epoch | Val NDCG@10 | Val Recall@10 | Val MRR@10 | Val HitRate@10 | D cosine NDCG@10 | Sure |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| D0 | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| D1 | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| D2 | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| D3 | ... | ... | ... | ... | ... | ... | ... | ... | ... |

- Her fold için uc seed'in ortalamasini hesapla; önce **fold bazli** matched cosine farkini raporla, sonra fold ortalamasini al.
- 9 neural kosusunu 9 bağımsız veri bolumu gibi yorumlama. Acikca fold/seed sapmalarini göster.
- Az sayida fold ve ortak kaynak graflari nedeniyle belirsizlik büyük olabilir; istatistiksel üstünluk iddiasi için daha bağımsız test gerekir.
- Başarıli denen bir varyantin sadece bir metriği az farkla gecmesini 'kesin ilerleme' olarak sunma.

### 7.4 Pratik karar kriterleri

- **Tercih edilebilir geliştirme adayi:** D0'a göre NDCG@10 artışi ve Recall@10'da belirgin bozulma olmaması; iyileşmenin birden fazla fold/seed'de görülmesi.
- **Güçlü aday:** Bunlara ek olarak paired D cosine baseline'a göre de NDCG@10 ve Recall@10 farklarinin tutarlı pozitif olması.
- **Yetersiz kanit:** Fark yalnız tek fold'a bagliysa, confidence/değişkenlik yuksekse veya iyileşme sadece seçilen validation'da bulunuyorsa genelleme iddiasi yok.
- **D1/D2 başarısizsa:** D3'un zorunlu olarak başarılı olacağı varsayilmaz. Daha büyük model yerine etiket güvenilirligi, audio kapsamı ve pozitif/negatif tanımı incelenir.
- **Bağımsız test:** Yeni lisansi/provenansi doğrulanmış ve eğitim sürecinden ayrı tutulan etiket/kohort bulunduğunda, karara baglanan modeli bir kez nihai değerlendir.

---

## 8. Kod tabani ve önerilen değişıklik noktaları

Asagidaki yollar onceki durum raporunda bildirilen mevcut dosyalardir. Bunlarin hepsinin bu oturumda kodu ayrıca incelenmedi; agent önce gerçek API'leri doğrulamali.

| Dosya | Görev / beklenen değişıklik |
|---|---|
| `src/triplets.py` | D1 sampling strategies, eligible candidates, deterministic fallback |
| `src/train_neurotune_v1.py` | D1 train embedding refresh, D2 train-only mask, telemetry |
| `src/model_v1.py` | Yalnizca ihtiyac varsa kontrollü branch dropout; eski checkpoint uyumlulugunu koru |
| `src/data/neurotune_v1_dataset.py` | Var olan ragged batch ve time padding davranisini koru |
| `src/features_neurotune_v1.py` | Train-only dönüşumlerin aynen kalmasini sağla |
| `src/data/neurotune_v1_abcd_cv_training.py` | Veri ve fold protokolunu değiştirmeden bagla |
| `scripts/run_neurotune_v1_full_modal_ablation.py` | Salt okunur referans; varsayilan eski kosuyu değiştirme |
| `scripts/run_neurotune_v1_training_improvements.py` | Uygulandı: izole D1 runner |
| `scripts/run_neurotune_v1_modality_dropout.py` | Uygulandı: D0 schedule'larını kullanan izole D2 runner |
| `tests/` | Sampling, masking, determinism, leakage, parity, reporting testleri |

Önerilen yeni sonuç klasoru:

```text
results/neurotune_v1_training_improvements/
  experiment.json
  d1_config.json
  training_curves.csv
  validation_metrics.csv
  mining_diagnostics.csv
  checkpoints/
  triplet_schedules/
```

Kaynak, split, checkpoint, schedule ve çıktı hash'leri `experiment.json` içinde tutulur; ayrı bir `hashes.json` dosyası üretilmez.

Runner hicbir kosulda eski `results/neurotune_v1_full_modal_ablation/`, `results/neurotune_v1_abcd_cv/` veya `results/neurotune_v1_d_modality_sensitivity/` ciktisina yazmamali.

---

## 9. Test listesi

- [ ] Mevcut D0 default davranisi değişmedi; referans checkpoint ve validation parity sağlandı.
- [x] Mining yalnız train track ve train embedding havuzuyla çalışiyor.
- [x] Ayni seed ve aynı input, aynı triplet schedule hash'ini üretiyor.
- [x] Bilinen train pozitifleri negatif secilmiyor.
- [x] Bos semi-hard havuzu uniform fallback ile güvenli sonuclaniyor.
- [x] Yanitilmis / bilinmeyen negatifler hakkinda rapor kesinlik iddiasi kurmuyor.
- [x] Train/validation/test artışt ayrımı korunuyor.
- [x] Yeni bağımsız test etiketleri train, mining veya validation kararlarinda kullanılmiyor.
- [ ] Audio padded timestep'leri masked kalmaya devam ediyor.
- [x] Modality dropout training disinda (validation/inference) kapali.
- [x] Dropout oranları, seed ve branch davranisi birim testlerle doğrulandı; gerçek koşu tanı satırlarıyla doğrulandı.
- [ ] Her D varyanti aynı validation query ve gallery evreninde karşılastiriliyor.
- [x] Eski dosyalarin hash'leri ve namespace'leri değişmedi.
- [x] Cikti raporunda tüm parametreler, run provenance ve seçilen epoch var.
- [x] `unittest`, `compileall`, `pip check` başarılı; gerçek uzun eğitim sonucu ayrıca audit edildi.

### Doğrulama komutlari

Proje kokunden, mevcut raporlarda kullanılan komutlar:

```bash
.venv/bin/python -m unittest discover -s tests -q
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/python -m pip check
```

D1 runner komutlari README'ye eklendi. Sabit protokol `config/neurotune_v1_d1_semi_hard.json` dosyasindan okunur. D1 fold-0 pilotu ve tam 3-fold × 3-seed eğitim tamamlandı; tam koşunun 450 epoch süresi `training_curves.csv` içinde bulunur. D2 runner komutu ve sabit protokolü README'ye eklendi; sentetik gerçek-trainer smoke testi, D2 koşusundan önce tamamlandı.

---

## 10. Agent'a verilecek görev özeti

1. **Mevcut repository'yi incele:** 17 Kasım D ablation ve modality sensitivity raporlarindaki protokol, model ve ciktilarin koddaki karşıligini tespit et.
2. **D0'u koru:** Bugun çalışan eski checkpoint ve rapor namespace'lerini değiştirme; default pipeline parity'sini test et.
3. **D1'i uygula:** Tamamlandı; train-only %75 uniform + %25 semi-hard negative mining, 3 warm-up epoch, 5 epoch'ta embedding yenileme, deterministik fallback ve ayrıntili audit (sonuçlar yukarıda).
4. **D2'yi uygula:** Tamamlandı. Training-only post-activation branch embedding dropout (`genre=.25`, `lyrics=.20`, `bpm=.10`, `audio=.00`) ve D0 ile eşleşmiş validation kontrolü; sonuçlar yukarıda kaydedildi.
5. **D3'u uygula:** Tamamlandı; D1+D2 bağımsız konfigürasyonla eğitildi, D1 checkpoint'i reuse/fine-tune edilmedi; sonuçlar 5.1'de.
6. **Önce unit/parity testleri**, sonra tek fold üzerinde kontrollü pilot, sonra protokol kilitlenince 3 fold × 3 seed. D3 için 501 test; D4 için 546 test, gerçek model/trainer sentetik smoke testi, cache preflight'i, pilot ve sabit 3×3×50 koşu tamamlandı.
7. **Raporla:** Recall@10, HitRate@10, MRR@10, paired cosine farklari, fold/seed varyasyonu, aktif triplet oranı veya window coverage, runtime ve provenance hash'leri. D3 ve D4 sonuçları README'ye; D4 tasarımı ve sonuçları bu planın 6. bölümüne kaydedildi.
8. **Data sınırini koru:** D1–D4'te D0'dan devralınan official/legacy test-source pozitifleri yalnız train-local negatif elemesi için kullanılabilir ve bu sınır açıkça raporlanır. Gelecekteki bağımsız test etiketleri mining/filtreleme dahil hiçbir aşamada görünmemeli. Tüm D1–D4 sonuçları validation-only geliştirme bulgularıdır.
9. **Karar ver:** D0–D4 validation sonuçlarını yalnız geliştirme bulguları olarak karşılaştır; validation'dan model üstünlüğü veya genelleme iddiası çıkarma. Yeni bağımsız test kohortu ve önceden belirlenmiş analiz olmadan nihai kazanan ilan etme.
10. **D4:** Tamamlandı; D0'a karşı izole multi-window audio karşılaştırmasıdır. D1/D2/D3 ile birleştirilmedi ve yalnız validation sonuçları üretti.

### Teslim edilecekler

- Calisan yeni strateji modulleri, konfigürasyonlar ve unit testleri.
- İzole deney runner'i ve değiştirilmemis eski artefaktlar.
- Fold/seed bazli JSON+CSV metrikleri, training curves ve mining diagnostics.
- Yeni `README`/teknik dokümantasyonda hem kazanımlar hem olumsuz bulgular ve etiket sınırları.
- **Yeni test açılmadıysa:** `independent_test_completed=false` gibi açık bir alan ve *bağımsız performans iddiasi yoktur* notu.

---

## 11. Nihai karar

**Son durum ve sonraki adım:** D1–D4 sabit validation-only geliştirme koşuları tamamlandı. D4'te modelin ortalama NDCG@10 değeri D0'dan +0.003882 yüksekti; fold ortalama farkları +0.004770 / +0.000460 / +0.006415 oldu. Ancak D4 modelinin genel ortalaması D4-matched cosine'dan düşüktü ve fold 1/2'de model cosine'ı geçmedi. D4 sonucu da model üstünlüğü veya genelleme kanıtı değildir. Bağımsız test çalıştırılmadı; yeni kohort, etiket kaynağı veya önceden belirlenmiş doğrulama protokolü ayrıca onaylanmadan başka eğitim koşusu başlatılmamalıdır.

Başarıli sonuç, tek bir validation skorunda artış değil; **D0 ve matched cosine baseline'a karşı aynı evaluation kosullarinda tutarlı iyileşme**, düzgün audit ve yeni bağımsız test ile ileride doğrulanabilecek bir modeldir. Sonuclar beklenenden düşuk kalirsa bu da final raporunda açıkça analiz edilmelidir.
