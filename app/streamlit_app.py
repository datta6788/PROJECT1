import streamlit as st
import pandas as pd
import joblib
import requests
from scipy.sparse import load_npz
from rapidfuzz import process 
st.markdown("""
<style>           
img{border-radius: 12px;transition: transform 0.3s ease;}
img:hover{transform: scale(1.05);}
</style>
""", unsafe_allow_html=True)

knn_model = joblib.load(r"D:\PRJ1\PROJECT1\models\knn_model.pkl")
sparse_matrix = load_npz(r"D:\PRJ1\PROJECT1\models\sparse_matrix.npz")
ani_to_index = joblib.load(r"D:\PRJ1\PROJECT1\models\ani_to_index.pkl")
index_to_ani = joblib.load(r"D:\PRJ1\PROJECT1\models\index_to_ani.pkl")
ind=joblib.load(r"D:\PRJ1\PROJECT1\models\ind.pkl")
top_sim=joblib.load(r"D:\PRJ1\PROJECT1\models\top_sim.pkl")
content_anime_index=joblib.load(r"D:\PRJ1\PROJECT1\models\content_anime_index.pkl")

def content_based_recc(anime_name,no_reccom=50):
    index=ind[anime_name]
    sim_scores=top_sim[index][:no_reccom]
    recomm=[]
    for ani_index, score in sim_scores:
        recomm.append({"Name":content_anime_index[ani_index],"score":float(score)})
    return pd.DataFrame(recomm)

def knn_recc(ani_name, no_recc=50):
    if ani_name not in ani_to_index:
        return None
    ani_index = ani_to_index[ani_name]
    distance, index = knn_model.kneighbors(sparse_matrix[ani_index],n_neighbors=no_recc + 1)
    recco = []
    for i in range(1, len(distance.flatten())):
        recommended_index = int(index.flatten()[i])
        recommended_anime = index_to_ani[recommended_index]
        distance_value = float(distance.flatten()[i])
        correct_distance_value=1-distance_value
        recco.append({"Name": recommended_anime,"Distance":correct_distance_value})
    return pd.DataFrame(recco)

def hybrid(ani_name,no_recomm=10,collab_weight=0.75,content_weight=0.25):
    content_dataframe=content_based_recc(ani_name,no_reccom=50)
    collab_dataframe=knn_recc(ani_name,no_recc=50)
    hybrid_dataframe=pd.merge(content_dataframe,collab_dataframe,on="Name",how="outer")
    hybrid_dataframe["score"]=hybrid_dataframe["score"].fillna(0)
    hybrid_dataframe["Distance"]=hybrid_dataframe["Distance"].fillna(0)
    hybrid_dataframe["hybrid_score"]=(collab_weight*hybrid_dataframe["Distance"]+content_weight*hybrid_dataframe["score"])
    hybrid_dataframe=hybrid_dataframe.sort_values(by="hybrid_score",ascending=False)
    # hybrid_dataframe=hybrid_dataframe.drop_duplicates(subset="Name")
    return hybrid_dataframe.head(no_recomm)

def fuzzy(input,anime_list):
    match=process.extractOne(input,anime_list)
    if match is None:
        return None
    return match[0]

def ani_data(ani_name):
    url=(f"https://api.jikan.moe/v4/anime?q={ani_name}&limit=1")
    response=requests.get(url)
    data=response.json()
    try:
        anime=data["data"][0]
        poster=anime["images"]["jpg"]["image_url"]
        score=anime["score"]
        episodes=anime["episodes"]
        genre=", ".join(genre["name"] for genre in anime["genres"])
        return{"poster":poster,"score":score,"episodes":episodes,"genre":genre}
    except:
        return None
def get_poster(anime_name):
    data=ani_data(anime_name)
    if data is not None:
        return data["poster"]
    return None

ani_list=list(ani_to_index.keys())
st.title("Anime RECOMMENDATION SYSTEM")
ani_input = st.text_input("Enter anime name")
top_n = st.selectbox("No.of recommendations",[5, 10, 25, 50, 100])
if st.button("Recommend"):
    matched_ani=fuzzy(ani_input,ani_list)
    hero_banner=ani_data(matched_ani)
    if hero_banner:
        column1,column2=st.columns([1,3])
        with column1:
            st.image(hero_banner["poster"],width=250)
        with column2:
            st.title(matched_ani.title())
            st.write(f"⭐ {hero_banner['score']}")
            st.write(f"🎬 {hero_banner['episodes']} Episodes")
            st.write(f"🎭 {hero_banner['genre']}")
        st.divider()
    # result = hybrid(matched_ani,top_n)
    content_result=content_based_recc(matched_ani,top_n)
    knn_result=knn_recc(matched_ani,top_n)
    hybrid_result=hybrid(matched_ani,top_n)
    if content_result is None or knn_result is None or hybrid_result is None:
        st.error("Anime not found!")
    else:
        #row1
        st.subheader(f"Anime similar to {ani_input}")
        cols=st.columns(5)
        for col,(_,row) in zip(cols,content_result.head(5).iterrows()):
            with col:
                poster=get_poster(row["Name"])
                if poster:
                    st.image(poster)
                st.caption(row["Name"].upper())
        #row2
        st.subheader(f"People also like")
        cols=st.columns(5)
        for col,(_,row) in zip(cols,knn_result.head(5).iterrows()):
            with col:
                poster=get_poster(row["Name"])
                if poster:
                    st.image(poster)
                st.caption(row["Name"].upper())
        #row3
        st.subheader("Top picks")
        cols=st.columns(5)
        for col,(_,row) in zip(cols,hybrid_result.head(5).iterrows()):
            with col:
                poster=get_poster(row["Name"])
                if poster:
                    st.image(poster)
                st.caption(row["Name"].upper())

