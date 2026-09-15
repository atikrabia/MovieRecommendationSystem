import streamlit as st
import pickle
import pandas as pd
import requests

# Sayfayı geniş ekran moduna alıyoruz
st.set_page_config(page_title="Film Tavsiye Sistemi", layout="wide")


# TMDB API'sinden afiş çeken fonksiyon
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


# Tavsiye fonksiyonu (Artık ID ve afişleri de döndürüyor)
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movie_names = []
    recommended_movie_posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id

        # Afiş fonksiyonunu çağırıp listeye ekliyoruz
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters


# Web sitesi tasarımı
st.title('🎬 Film Tavsiye Sistemi')

# Pickle dosyalarını yüklüyoruz
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Film seçme kutusu
selected_movie_name = st.selectbox(
    'Bana sevdiğin bir filmi söyle, sana benzerlerini posterleriyle önereyim...',
    movies['title'].values
)

# Tavsiye Et butonu basıldığında
if st.button('Tavsiye Et'):
    names, posters = recommend(selected_movie_name)

    # 5 sütun oluşturup filmleri yan yana kartlar halinde diziyoruz
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])