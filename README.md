# 🎬 Film Tavsiye Sistemi (Movie Recommendation System)

Bu proje, kullanıcıların sevdiği filmlere dayanarak onlara matematiksel olarak en çok benzeyen diğer filmleri öneren **İçerik Tabanlı (Content-Based)** bir makine öğrenmesi modelidir.

## 🛠️ Kullanılan Teknolojiler
* **Python**
* **Pandas** (Veri manipülasyonu ve ön işleme)
* **AST** (Metin formundaki karmaşık veri yapılarını listelere dönüştürme)

###############################################################
# 7. 'overview' (özet) sütunundaki uzun cümleyi de kelime kelime ayırıp listeye çeviriyoruz
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# 8. İsimlerdeki ve türlerdeki boşlukları silme işlemi (Örn: "Science Fiction" -> "ScienceFiction")
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# 9. Tüm bu listeleri 'tags' (etiketler) adında devasa tek bir sütunda birleştiriyoruz
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# 10. Artık eski sütunlara ihtiyacımız kalmadı. Sadece ID, Başlık ve Tags'i alıyoruz
new_df = movies[['movie_id', 'title', 'tags']].copy()

# 11. Son rötüş: 'tags' içindeki listeyi tekrar boşluklu düz bir metne çevirip, her şeyi küçük harf yapıyoruz
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Hazır olan yeni veri setimizin ilk filmine (Avatar) ait devasa metni görelim
print(new_df['tags'][0])
####################################################
**BU KOD SATIRLARINDA NE YAPILDIĞI AŞAĞIDA ANLATILMIŞTIR:**

## 🧠 Özellik Mühendisliği (Feature Engineering) Adımlarının Mantığı

Metin tabanlı verilerimizin, makine öğrenmesi algoritması tarafından doğru bir şekilde okunup işlenebilmesi için şu 4 temel veri dönüşüm adımı uygulanmıştır:

### 1. Parçala (Split İşlemi)
Filmin özeti (`overview`) uzun bir cümle formatındaydı ("Bir zamanlar uzak bir galakside..."). Ancak oyuncular ve türler gibi diğer sütunlarımız liste halindeydi (`['Aksiyon', 'Macera']`). İleride bunları sağlıklı bir şekilde uç uca ekleyebilmek için, uzun özet cümlesi boşluklarından kesilerek liste formuna dönüştürülmüştür (`['Bir', 'zamanlar', 'uzak', 'bir', 'galakside...']`).

### 2. İsimleri Tek Bir Etikete Dönüştürme (Boşlukları Silme)
Modelin matematiksel doğruluğu için **en kritik** adımdır. 
Örneğin elimizde iki farklı film olsun:
* İlk filmin yönetmeni: **Sam Mendes**
* İkinci filmin yönetmeni: **Sam Raimi**

Eğer kelimelerin arasındaki boşluklar silinmezse, algoritma metinleri boşluklardan bölerek okuyacağı için "Sam" kelimesini iki film için de ortak bir değişken olarak algılar ve filmleri birbirine haksız yere %50 oranında benzer bulur. 
Boşluklar silinip `SamMendes` ve `SamRaimi` formatına getirildiğinde ise algoritma bunları tıpkı birer hashtag (#) gibi **tek ve eşsiz birer etiket** olarak görür.
İsim benzerliğinden doğan karışıklık tamamen önlenmiş olur.

### 3. Tek Bir Metin Havuzu Oluşturma (Tags Sütunu)
Algoritmanın sütunları (oyuncular, türler, özet vb.) ayrı ayrı kategorize etmesine gerek yoktur. 
Modelin ihtiyacı olan tek şey, o filmi temsil eden devasa bir kelime havuzudur. 
Bu nedenle tüm ayrıştırılmış ve temizlenmiş metinler **`tags`** adında tek bir ana sütunda birleştirilmiştir.

### 4. Standartlaştırma (Küçük Harf Dönüşümü)
Makine öğrenmesi modelleri büyük/küçük harf duyarlıdır (Case Sensitive). 
Yani algoritma için `Action` kelimesi ile `action` kelimesi birbirinden tamamen farklı iki değişkendir.
Bu frekans hatasını önlemek için tüm metinler küçük harfe (`.lower()`) çevrilmiş ve liste yapısından çıkarılıp aralarında birer boşluk olan düz bir cümleye dönüştürülmüştür.

> **💡 Özetle:** Bu adımların sonucunda, makine öğrenmesi modelinin okuyup **kelime frekanslarını (hangi kelimeden kaç tane geçtiğini)** kusursuzca sayabileceği, pürüzsüz ve standartlaştırılmış bir metin bloğu inşa edilmiştir.

"overview" parçalanması gereken uzun bir cümleydi, onu kelimelerine böldük.
"diğerleri" bütün kalması gereken özel isimlerdi, onları birbirine yapıştırdık.
Günün sonunda hepsini aynı formata (bir kelime listesine) getirmiş olduk ki, aralarına + işareti koyduğumuzda pürüzsüzce uç uca eklenebilsinler.



############################################################
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Vektörizasyon İşlemi
# max_features=5000: En sık geçen 5000 kelimeyi al.
# stop_words='english': İngilizce bağlaçları yoksay.
cv = CountVectorizer(max_features=5000, stop_words='english')

# Metinlerimizi sayılardan oluşan devasa bir matrise (vektörlere) dönüştürüyoruz
vectors = cv.fit_transform(new_df['tags']).toarray()

# 2. Kosinüs Benzerliği Matrisini Oluşturma
# Her filmin diğer 4806 filmle olan benzerlik açısını hesaplıyoruz
similarity = cosine_similarity(vectors)

# Benzerlik matrisinin boyutunu ve Avatar filminin (ilk filmin) diğer filmlerle benzerlik skorlarını görelim
print("\n--- BENZERLİK MATRİSİ BOYUTU ---")
print(similarity.shape)

print("\n--- AVATAR FİLMİNİN DİĞER FİLMLERLE BENZERLİK SKORLARI (İlk 10 Film) ---")
print(similarity[0][:10])
###################################################################
**BU KOD SATIRLARINDA NE YAPILDIĞI AŞAĞIDA ANLATILMIŞTIR:**

## 🔢 Doğal Dil İşleme (NLP) ve Vektörizasyon

Makine öğrenmesi algoritmaları metinleri doğrudan işleyemez.
Bu nedenle metin tabanlı verilerin sayısal bir matrise dönüştürülmesi gerekir.
Kelimeleri matematiksel vektörlere çevirmek için `scikit-learn` kütüphanesi kullanılmıştır.
Bu aşamada **CountVectorizer** aracı ve **Kosinüs Benzerliği (Cosine Similarity)** tercih edilmiştir.

Süreç şu üç temel adımda gerçekleşmiştir:

### 1. Değişkenlerin (Sütunların) Belirlenmesi
* Algoritma öncelikle tüm veri setindeki kelimeleri tarar.
* Anlamsız İngilizce bağlaçları (`stop_words='english'`) temizler.
* Kalan havuz içinden en sık tekrar eden **5000** kelimeyi (`max_features=5000`) seçer.
* Bu kelimeleri boş bir tablonun sütun başlıkları (bağımsız değişkenler) olarak konumlandırır.

### 2. Frekans Matrisinin Doldurulması (Satırlar)
* Elimizde 4806 film (satır) ve 5000 kelime (sütun) barındıran devasa bir matris oluşur.
* Her film için tek tek kelime sayımı yapılır.
* Filmin etiket havuzunda ilgili kelime kaç kez geçiyorsa, o hücreye frekans değeri yazılır.

### 3. Kosinüs Benzerliği ile Eşleştirme
* İşlem sonucunda metin formundaki her film, 5000 sayıdan oluşan bir satıra (vektöre) dönüşür.
* Kosinüs Benzerliği algoritması, 5000 boyutlu bu uzayda her filmi bir nokta olarak çizer.
* Vektörler arasındaki **açıları hesaplayarak** sayısal örüntüsü birbirine en çok benzeyen filmleri tespit eder.




❌ Enumerate Olmadan Ne Olurdu?
Diyelim ki elinizde 5 kişilik bir sınıfın sınav notları var. Öğretmen notları tahtaya öğrencilerin okul numarasına (sırasına) göre yazmış:

0. Öğrenci (Ali): 40 aldı

1. Öğrenci (Ayşe): 90 aldı

2. Öğrenci (Can): 50 aldı

3. Öğrenci (Ece): 95 aldı

4. Öğrenci (Cem): 60 aldı

Elindeki not listesi şu: [40, 90, 50, 95, 60]

Eğer bilgisayara sadece "Bu listeyi büyükten küçüğe sırala" dersen, bilgisayar sayıları alır ve şöyle dizer:
👉 [95, 90, 60, 50, 40]

Harika, en yüksek notun 95 olduğunu bulduk. Ama bu 95'i kim aldı? Sıralamayı değiştirdiğin an, sayıların baştaki sırası (kimlikleri) tamamen kayboldu. Ali mi 95 aldı, Ece mi bilemezsin. Sadece 95 diye havada asılı duran bir sayı kalır elinde.

✅ Enumerate İle Ne Yapıyoruz?
İşte enumerate, biz bu notları (skorları) sıralayıp sırasını bozmadan önce devreye girer. Her notun üzerine, o notun sahibinin sıra numarasını bir etiket gibi yapıştırır. Listeyi paketler haline getirir:
👉 [ (0, 40), (1, 90), (2, 50), (3, 95), (4, 60) ]

Şimdi bilgisayara "Bu paketleri notlara göre büyükten küçüğe sırala" dediğimizde, o parantezler hiç bozulmadan blok halinde yer değiştirir:
👉 [ (3, 95), (1, 90), (4, 60), (2, 50), (0, 40) ]

Artık en yüksek notun 95 olduğunu ve bu notun 3 numaralı kişiye ait olduğunu kaybetmemiş olduk. Gidip 3 numaralı kişiyi listeden bulabiliriz.


