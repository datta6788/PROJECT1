import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import load_npz

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
        recco.append({"Anime": recommended_anime,"Distance": round(distance_value,4)})
    return recco

st.title("Anime RECOMMENDATION SYSTEM")
ani_input = st.text_input("Enter anime name")
top_n = st.selectbox("No.of recommendations",[5, 10, 25, 50, 100])
if st.button("Recommend"):
    result = knn_recc(ani_input,top_n)
    if result is None:
        st.error("Anime not found!")
    else:
        st.dataframe(result,use_container_width=True)