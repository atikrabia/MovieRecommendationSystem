import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

# 1. Verileri okuma
movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')

print("--- MOVIES VERİ SETİ ---")
print(movies.head(2))
print("\n--- CREDITS VERİ SETİ ---")
print(credits.head(2))

# 2. İki tabloyu 'title' (film adı) sütunu üzerinden birleştirme
movies = movies.merge(credits, on='title')

# 3. Model için sadece gerekli olan sütunları seçme (Feature Selection)
# Gerekli olanlar: id, başlık, özet, türler, anahtar kelimeler, oyuncular ve ekip (yönetmen için)
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

print(movies.head(2))

# 2. Eksik (Null) olan o 3 satırı tablodan silme
movies.dropna(inplace=True)

print("\n--- EKSİK VERİ KONTROLÜ ---")
print(movies.isnull().sum())

# Verinin son halini yeni bir dosya olarak kaydeder
movies.to_csv("temizlenmis_veri.csv", index=False)


# 3. Karmaşık sözlük yapısından sadece "isimleri" çeken bir fonksiyon
def convert(text):
    L = []
    # ast.literal_eval metin (string) olarak gelen yapıyı Python listesine çevirir
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

# 4. Bu fonksiyonu türler (genres) ve anahtar kelimeler (keywords) sütunlarına uygulama
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

print(movies.head(2))
movies.to_csv("temizlenmis_veri2.csv", index=False)


# 5. Oyuncular (cast) içinden sadece ilk 3 başrolü alan fonksiyon
def convert3(text):
    L = []
    sayac = 0
    for i in ast.literal_eval(text):
        if sayac < 3: # Sadece ilk 3 kişiyi al
            L.append(i['name'])
            sayac += 1
        else:
            break # 3 kişiyi aldıktan sonra dur
    return L

movies['cast'] = movies['cast'].apply(convert3)


# 6. Ekip (crew) içinden sadece 'Yönetmen'i bulan fonksiyon
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director': # Eğer görevi Yönetmen ise
            L.append(i['name'])    # İsmini listeye ekle
            break                  # Yönetmeni bulunca aramayı bırak
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# Tablonun son haline göz atalım
print(movies.head(3))
movies.to_csv("temizlenmis_veri3.csv", index=False)

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
movies.to_csv("temizlenmis_veri4.csv", index=False)



# 1. Vektörizasyon İşlemi
# max_features=5000: En sık geçen 5000 kelimeyi al.
# stop_words='english': İngilizce bağlaçları yoksay.
cv = CountVectorizer(max_features=5000, stop_words='english')

# Metinlerimizi sayılardan oluşan devasa bir matrise (vektörlere) dönüştürüyoruz
vectors = cv.fit_transform(new_df['tags']).toarray()

# 2. Kosinüs Benzerliği Matrisini Oluşturma
# Her filmin diğer 4806 filmle olan benzerlik açısını hesaplıyoruz
# Kosinüs benzerliğini hesaplayıp float32 yaparak boyutu yarıya düşürüyoruz
similarity = cosine_similarity(vectors).astype(np.float32)


# Benzerlik matrisinin boyutunu ve Avatar filminin (ilk filmin) diğer filmlerle benzerlik skorlarını görelim
print("\n--- BENZERLİK MATRİSİ BOYUTU ---")
print(similarity.shape)

print("\n--- AVATAR FİLMİNİN DİĞER FİLMLERLE BENZERLİK SKORLARI (İlk 10 Film) ---")
print(similarity[0][:10])


def recommend(movie):
    # 1. Kullanıcının girdiği filmin tablodaki indeks (satır) numarasını buluyoruz
    movie_index = new_df[new_df['title'] == movie].index[0]

    # 2. O filme ait kosinüs benzerlik skorlarını matristen çekiyoruz
    distances = similarity[movie_index]

    # 3. Skorları sıralarken HANGİ filme ait olduklarını (indekslerini) kaybetmemek için 'enumerate' kullanıyoruz.
    # Tersine (büyükten küçüğe) sıralayıp, kendisi hariç ilk 5 filmi [1:6] alıyoruz.
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    # 4. Bulduğumuz o 5 filmin indeks numaralarını kullanarak isimlerini ekrana yazdırıyoruz
    print(f"\n--- '{movie}' SEVENLER BUNLARI DA SEVDİ ---")
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# Sistemi test edelim!
recommend('Batman Begins')
recommend('Avatar')


# new_df tablosunu ID'leri de içerecek şekilde bir sözlüğe çevirip kaydediyoruz
movies_dict = {
    'movie_id': new_df['movie_id'].values, # Film ID'lerini buraya ekledik
    'title': new_df['title'].values
}
pickle.dump(movies_dict, open('movie_dict.pkl', 'wb'))

# Kosinüs matrisini aynen kaydetmeye devam ediyoruz
pickle.dump(similarity, open('similarity.pkl', 'wb'))




