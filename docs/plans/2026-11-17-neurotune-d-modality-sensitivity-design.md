# D Modeli Tek-Modalite Validation Duyarlılığı — Tasarım

**Tarih:** 17 Kasım 2026  
**Durum:** Uygulandı; validation-only koşu ve son doğrulama tamamlandı.  
**Kapsam:** A–D CV katalog ve fold'ları üzerinde, eğitilmiş dokuz D checkpoint'i ile yalnızca validation galerisi.

## Karar

Modeli yeniden eğitmeden D'nin audio/BPM/genre/lyrics girdilerinden birini tek tek sıfırladığımızda davranışını ölç. Başka kombinasyon, hiperparametre araması, checkpoint seçimi veya test değerlendirmesi yapma. Maskesiz referans, tamamlanmış P0 ablation raporundaki aynı fold/seed validation metrikleriyle eşleşmelidir; aksi halde yayınlama.

## Veri akışı ve sınır

Var olan `load_abcd_cv_training` ile kaynağın/split'in doğrulandığı validation-only P0 yolunu yeniden kullan; MXM kaynağını katalog kimlikleriyle sınırla. Her fold için `build_abcd_cv_development_features` yalnız train+validation şarkılarını getirir ve dönüşümleri yalnız train ID'lerinde fit eder. Validation şarkıları track ID ile sıralanır; eski test cohort'unun ID'lerini, feature'larını, embedding'lerini, relevance'ını veya metriklerini yeni analiz oluşturmaz. Loader'ın gerekli test-source hash ve bilinen-pozitif negatif-hariç-tutma provenance kontrolleri hedef etiketi olarak kullanılmaz.

P0 raporundaki D checkpoint path/hash, fold/seed, variant, selected epoch, transform/split/schedule hash ve seçili validation metrikleri dosya baytlarıyla karşılaştırılmalı. Eski C `experiment.json` veya C test metrikleri açılmamalıdır. Orijinal maskesiz validation sıralama ve metrikleri raporla eşleşmeden maske sonuçları kabul edilmez. Read-only inputlar değişirse yalnız yeni geçici çıktı temizlenir; hiçbir eski artifact silinmez veya üzerine yazılmaz.

## Sabit deney koşulları

Her D checkpoint'i için beş koşul: `original`, `audio`, `bpm`, `genre`, `lyrics`. Bir koşulda tek modalite tüm validation şarkılarında **model girdisi olarak** sıfırlanır; kalanlar değişmez. Audio'da yalnız `(T,26)` değerleri sıfır olur; ragged uzunluklar ve boolean padding mask aynı kalır. Diğer dallarda train-fitted float32 dizisi sıfırlanır. Model `eval()` ve `no_grad()` modunda tutulur; ağırlık, optimizer, eğitim schedule'ı, train-only istatistikleri ve etiketler değiştirilmez. Maskesiz girdi nesneleri mutasyona uğratılmaz.

Sabit validation query/gallery ve bilinen validation positives üzerinden tüm eligible query'ler için mevcut k=10 self-excluding, track-ID tie-breaking retrieval protokolünü kullan. Her koşulda Recall@10, HitRate@10, MRR@10, NDCG@10, eligible query ve positive edge sayılarını kaydet. Orijinal sıralamaya göre query bazında Top-10 küme örtüşme oranı; her şarkı için orijinal ile maskeli L2-normalized embedding cosine benzerliği/kayması hesapla. Değişimlerin ortalaması fold/seed bazında kaydedilir; ayrıca raporda 9 gözlemin açıklayıcı ortalamaları verilir. İsteğe bağlı maskesiz kontrol için sıralama çıktılarından kendi hesaplanan orijinal metriği P0 raporuyla karşılaştır; float toleransı küçük ve sabit olmalı.

**Yorum sınırı:** Sıfırlama eğitim dağılımının dışında bir müdahaledir; yüksek kayma o modalitenin model tarafından kullanıldığına dair duyarlılık sinyali olabilir, nedensel katkı veya genel başarı kanıtı değildir. Özellikle audio sıfırı normalizasyon ortalamasını temsil ederken mask/uzunluk değişmez; türde sıfır all-off kodu ve lyrics'te sıfır boş dokümandır. Top-10 ölçümleri validation checkpoint seçimiyle aynı etiketleri kullandığından yeni performans iddiası doğurmaz.

## Çıktı ve doğrulama

İzole, daha önce var olmayan `results/neurotune_v1_d_modality_sensitivity/` alanına manifest/JSON ve küçük CSV özetleri yayınla; P0/A–D CV namespace'lerine yazma. Kaynak, kod, P0 manifest ve checkpoint SHA-256; fold, seed, mask, metrik, sorgu/kenar sayısı, mean embedding cosine ve mean Top-10 overlap kaydedilir. Staging, no-overwrite ve hata durumunda rollback korunur.

Sentetik testler maskenin yalnız doğru tensörü değiştirdiğini ve orijinal girdiyi bozmadığını; audio padding mask/uzunluk sabitliğini; tüm şarkıları maskeyi; original endpoint parity zorunluluğunu; exact fold/seed/hash preflight'ını; test feature/label erişimi olmadan çalışmayı; output collision/rollback'ı ve tie determinism'i doğrular. Önce focused testler, sonra tek gerçek validation-only koşu, artifact provenance ve tam suite/compileall/pip check. Sonuçlar Türkçe, açıkça tanısal ve validation-only olarak belgelenir.
