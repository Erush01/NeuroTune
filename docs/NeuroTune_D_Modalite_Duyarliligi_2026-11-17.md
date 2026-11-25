# NeuroTune D Modeli: Tek-Modalite Duyarlılığı

**Tarih:** 17 Kasım 2026  
**Statü:** Tamamlanmış validation-only tanısal analiz; yeni test değerlendirmesi değildir.  
**Rapor:** `results/neurotune_v1_d_modality_sensitivity/experiment.json`

## Soru ve yöntem

Eğitilmiş D modeli audio, BPM, genre ve lyrics girdilerinden herhangi biri sıfırlandığında nasıl davranıyor? Dokuz **sabit** D checkpoint'i (3 artist fold × 3 seed) yeniden eğitilmeden değerlendirildi. Her checkpoint için maskesiz referans ve dört tek-modalite maskesi çalıştırıldı; toplam **45** fold/seed/koşul kaydı var. Maske validation galerisinin bütün şarkılarına uygulanır. Audio zaman maskesi ve dizi uzunluğu sabit kalır. Her fold'da dönüşümler yalnız train track'lerinde fit edilir; kullanılan şarkı özellikleri ve pozitif kenarlar **yalnızca validation** içindir.

Her maskesiz koşunun Recall@10, HitRate@10, MRR@10, NDCG@10, sorgu ve pozitif kenar sayıları P0 raporunun aynı D fold/seed seçili-checkpoint validation kaydıyla eşleşmedikçe runner maskeli sonucu kabul etmez. Çıktılar ayrı namespace'e yayımlandı; P0 ve eski C checkpoint'leri değiştirilmedi. Test-source materyali yalnız mevcut input/provenans politikası gereği loader tarafından doğrulanabilir; test şarkısı için feature/embedding/relevance/metrik oluşturulmadı.

İki ek davranış ölçüsü: **Top-10 örtüşmesi**, maskesiz ve maskeli sıralamaların aynı sorgu için önerdiği şarkı kümelerinin ortak oranının eligible sorgular üzerindeki ortalaması; **embedding cosine kayması**, maskesiz/maskeli normalize şarkı embedding'leri için ortalama `1 − cosine` değeridir. Örtüşme 1 ise Top-10 kümesi değişmedi. Kayma 0 ise embedding aynı yöndedir; bu iki ölçü retrieval kalitesinin yerine geçmez.

## Sonuçlar

Aşağıdaki sayılar **dokuz fold-seed gözleminin aritmetik ortalamasıdır**; gözlemler birbirinden bağımsız test setleri değildir. NDCG farkı, maskesiz NDCG@10 eksi koşul NDCG@10 değeridir (yüksek pozitif değer daha büyük düşüş).

| Girdi koşulu | Recall@10 | HitRate@10 | MRR@10 | NDCG@10 | NDCG düşüşü | Top-10 örtüşmesi | Embedding cosine kayması |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Maskesiz D | 0.119240 | 0.194019 | 0.072913 | 0.071530 | 0 | 1.000000 | 0.000000 |
| Audio sıfır | 0.107578 | 0.176365 | 0.064955 | 0.063567 | 0.007963 | 0.586110 | 0.043439 |
| BPM sıfır | 0.119860 | 0.193007 | 0.070278 | 0.070262 | 0.001268 | 0.861922 | 0.003830 |
| Genre sıfır | 0.038017 | 0.065997 | 0.023998 | 0.022969 | 0.048561 | 0.265619 | 0.344550 |
| Lyrics sıfır | 0.092716 | 0.158442 | 0.053530 | 0.052619 | 0.018911 | 0.124028 | 0.329985 |

**Tanısal yorum:** Bu müdahalede en büyük NDCG düşüşü genre sıfırlandığında görüldü; ardından lyrics, audio ve BPM geliyor. Lyrics maskesi Top-10 öneri kümelerini en çok değiştiriyor (örtüşme yaklaşık `0.124`), fakat genre maskesi gözlenen validation pozitifleriyle eşleşen sıralamayı daha fazla bozuyor. Audio maskesi NDCG'yi `0.0080` kadar düşürüyor: audio kolunun etkisi sıfır değil, ama bu deneyde genre maskesinin etkisinden küçük. BPM maskesinin NDCG etkisi küçük; Recall@10 ortalamasının BPM maskesinde çok az yükselmesi metriklerin aynı yönde hareket etmesini gerektirmediğini gösteriyor. Fold düzeyinde BPM maskesinin NDCG'si fold 0'da maskesiz durumdan az yüksek (`0.089820` ve `0.088968`), diğer iki fold'da düşük; bu ayrıntı ortalamaya bakarak evrensel sonuç çıkarmamayı gerektirir.

**Sınır:** Sıfır genre, gerçek bir tür etiketi değil “hiçbir tür açık değil” kodudur; lyrics sıfırı boş dokümandır; audio sıfırı train-normalize kanallarda ortalamayı temsil ederken segment uzunlukları korunur. Bunlar eğitim dağılımı dışında müdahalelerdir. Sonuçlar modelin bu girdilere **duyarlılığını** gösterir, türün/lyrics'in/audio'nun nedensel katkısını veya şarkı benzerliğinin gerçek dünyadaki önem sırasını kanıtlamaz. Aynı validation NDCG checkpoint seçiminde zaten kullanıldı; bu sonuçlarla yeni checkpoint seçilmedi ve test seti açılmadı.

## Çıktılar ve doğrulama

- `results/neurotune_v1_d_modality_sensitivity/experiment.json`: protokol, 45 kayıt, açıklayıcı aggregate, input/kod/checkpoint hash'leri, açık test-erteleme bayrağı.
- `results/neurotune_v1_d_modality_sensitivity/condition_records.csv`: aynı 45 koşul kaydının satır bazında görünümü.
- Gerçek validation koşusu `.venv/bin/python -m scripts.evaluate_neurotune_v1_d_sensitivity` komutuyla **exit 0** bitti; yaklaşık 134 saniye sürdü. Her checkpoint'te maskesiz P0 validation parity kontrolü ve kaynak/checkpoint bütünlük kontrolleri geçti. Yeni rapor/CSV 45 benzersiz fold/seed/koşul kaydı içeriyor; dokuz checkpoint SHA-256 değeri karşılaştırıldı. Önceden var olan çıktılara yazılmadı.
- Yeni runner ve yardımcıların sentetik testleri: 11 runner testi ve 6 saf ölçüm testi geçti. Nihai doğrulama: `.venv/bin/python -m unittest discover -s tests -q` **440 test, exit 0**; `.venv/bin/python -m compileall -q src scripts tests` **exit 0**; `.venv/bin/python -m pip check` **exit 0**, `No broken requirements found.`

Mevcut sonuç namespace'i üzerine yazma reddedilir. Kod, tasarım ve uygulama adımları `scripts/evaluate_neurotune_v1_d_sensitivity.py`, `src/evaluate_d_modality_sensitivity.py` ve `docs/plans/2026-11-17-neurotune-d-modality-sensitivity-{design,implementation}.md` dosyalarındadır.
