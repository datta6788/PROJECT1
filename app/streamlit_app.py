import streamlit as st
import pandas as pd
import joblib
import requests
from scipy.sparse import load_npz
from rapidfuzz import process 

knn_model = joblib.load(r"D:\PRJ1\PROJECT1\models\knn_model.pkl")
sparse_matrix = load_npz(r"D:\PRJ1\PROJECT1\models\sparse_matrix.npz")
ani_to_index = joblib.load(r"D:\PRJ1\PROJECT1\models\ani_to_index.pkl")
index_to_ani = joblib.load(r"D:\PRJ1\PROJECT1\models\index_to_ani.pkl")

def knn_recc(ani_name, no_recc=10):
    if ani_name not in ani_to_index:
        return None
    ani_index = ani_to_index[ani_name]
    distance, index = knn_model.kneighbors(sparse_matrix[ani_index],n_neighbors=no_recc + 1)
    recco = []
    for i in range(1, len(distance.flatten())):
        recommended_index = int(index.flatten()[i])
        recommended_anime = index_to_ani[recommended_index]
        distance_value = float(distance.flatten()[i])
        recco.append({"Anime": recommended_anime})
    return pd.DataFrame(recco)

def fuzzy(input,anime_list):
    match=process.extractOne(input,anime_list)
    if match is None:
        return None
    return match[0]

def posters(ani_name):
    url=(f"https://api.jikan.moe/v4/anime?q={ani_name}&limit=1")
    response=requests.get(url)
    data=response.json()
    try:
        url=data["data"][0]["images"]["jpg"]["image_url"]
        return url
    except:
        return None

ani_list=list(ani_to_index.keys())
st.title("Anime RECOMMENDATION SYSTEM")
ani_input = st.text_input("Enter anime name")
top_n = st.selectbox("No.of recommendations",[5, 10, 25, 50, 100])
if st.button("Recommend"):
    matched_ani=fuzzy(ani_input,ani_list)
    result = knn_recc(matched_ani,top_n)
    if result is None:
        st.error("Anime not found!")
    else:
        st.success(f"Showing results for: {matched_ani}")
        for _,row in result.iterrows():
            url=posters(row["Anime"])
            Pcolumn,Acolumn=st.columns([1,3])
            with Pcolumn:
                if url:
                    st.image(url,width=150)
            with Acolumn:
                st.subheader(row["Anime"])
            st.markdown("---")