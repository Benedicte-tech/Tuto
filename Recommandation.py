import pandas as pd
import numpy as np
import json
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

st.title("Films français antérieurs à 2000")

st.write("Suivez nos recommandations de cinéphiles")

df = pd.read_csv('C:/Users/admin/Downloads/films_fr_avant2000_recommandation.csv')
options = df['title'].tolist()
options1 = options[::4]

# Afficher le selectbox avec les options lues
st.markdown("""
    <style>
        .selectbox-label {
            font-size: 30px;
            color: red;
            font-weight: 900;
            #margin-bottom:-40px;
        }
    </style>
""", unsafe_allow_html=True)

# Afficher le texte du label avec le style appliqué
st.markdown('<p class="selectbox-label">Choisissez un film que vous aimez:</p>', unsafe_allow_html=True)

# Afficher le selectbox
option = st.selectbox("", options1)

# Afficher le film choisi
#st.write("Vous avez choisi : ", option)
# option = st.selectbox("Choisissez un film que vous aimez", options1)

# st.write("Vous avez choisi: ", option)
scaler  = MinMaxScaler()
df[['popularity','Action', 'Animation', 'Aventure', 'Comédie', 'Crime', 'Documentaire', 'Drame', 'Familial', 'Fantastique', 'Guerre', 'Histoire', 'Horreur', 'Musique', 'Mystère', 'Romance', 'Science-Fiction', 'Thriller', 'Western']] = scaler.fit_transform(df[['popularity','Action', 'Animation', 'Aventure', 'Comédie', 'Crime', 'Documentaire', 'Drame', 'Familial', 'Fantastique', 'Guerre', 'Histoire', 'Horreur', 'Musique', 'Mystère', 'Romance', 'Science-Fiction', 'Thriller', 'Western']])

condition  =  df.title.str.contains(option, case = False)
df_ami = df[condition]
df_ami = df_ami.reset_index(drop=True)
X_ami = df_ami[['popularity','Action',
       'Animation', 'Aventure', 'Comédie', 'Crime', 'Documentaire', 'Drame',
       'Familial', 'Fantastique', 'Guerre', 'Histoire', 'Horreur', 'Musique',
       'Mystère', 'Romance', 'Science-Fiction', 'Thriller', 'Western']]

df_entrainement =  df[~condition]
df_entrainement  = df_entrainement.reset_index(drop=True)
X_entrainement  =  df_entrainement[['popularity','Action',
       'Animation', 'Aventure', 'Comédie', 'Crime', 'Documentaire', 'Drame',
       'Familial', 'Fantastique', 'Guerre', 'Histoire', 'Horreur', 'Musique',
       'Mystère', 'Romance', 'Science-Fiction', 'Thriller', 'Western']]

knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X_entrainement)
distances, indices = knn.kneighbors(X_ami)

# recommended_titles = df_entrainement.iloc[indices[0], :5]['title'].tolist()
# st.markdown('<p style="font-size:24px;color:red;font-weight:900;">Films recommandés :</p>', unsafe_allow_html=True)
# for title in recommended_titles:
#    st.write(title)


recommended_indices = indices[0]
recommended_titles = df_entrainement.iloc[recommended_indices]['title'].tolist()
recommended_overviews = df_entrainement.iloc[recommended_indices]['overview'].tolist()
recommended_lien_photos = df_entrainement.iloc[recommended_indices]['lien_photos'].tolist()
#recommended_genres = df_entrainement.iloc[recommended_indices]['genres'].tolist()

st.markdown('<p style="font-size:30px;color:red;font-weight:900;margin-top:60px;">Nous vous recommandons :</p>', unsafe_allow_html=True)

for title, overview,lien_photos in zip(recommended_titles, recommended_overviews,recommended_lien_photos):
    
    st.markdown(f"""
    <p style="text-align:center;font-weight:900;margin-top:30px;background-color: #eee; font-size: 20px; border: 1px solid #ccc; padding: 5px;">
        {title}
    </p>
""", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
          st.image(f"https://media.themoviedb.org/t/p/w300_and_h450_bestv2/{lien_photos}")
    with col2:
          st.markdown(f"""<p style="text-align:justify;">{overview}</p>""", unsafe_allow_html=True)
    
    
