import streamlit as st
import pickle
import pandas as pd
import requests
from collections import Counter
from rank_bm25 import BM25Okapi
from sklearn.metrics import jaccard_score

# ========== Fetch Movie Poster ==========
def fetch_poster(movie_id):
    response = requests.get(
        f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=1e0a3c35418718f52f86acfc6ec3200e&language=en-US')
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

# ========== Load Pickle Files ==========
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
cosine_sim_tfidf = pickle.load(open('cosine_sim_tfidf.pkl', 'rb'))
cosine_sim_lsi = pickle.load(open('cosine_sim_lsi.pkl', 'rb'))
cosine_sim_word2vec = pickle.load(open('cosine_sim_word2vec.pkl', 'rb'))
binary_vectors = pickle.load(open('binary_vectors.pkl', 'rb'))

# Load BM25 Tokenized Corpus
tokenized_tags = pickle.load(open('bm25_corpus.pkl', 'rb'))
bm25 = BM25Okapi(tokenized_tags)

# ========== Individual Recommendation Functions ==========
def get_recommendations_tfidf(movie):
    idx = movies[movies['title'] == movie].index[0]
    sim_scores = sorted(list(enumerate(cosine_sim_tfidf[idx])), key=lambda x: x[1], reverse=True)[1:6]
    return [movies.iloc[i[0]].title for i in sim_scores]

def get_recommendations_lsi(movie):
    idx = movies[movies['title'] == movie].index[0]
    sim_scores = sorted(list(enumerate(cosine_sim_lsi[idx])), key=lambda x: x[1], reverse=True)[1:6]
    return [movies.iloc[i[0]].title for i in sim_scores]

def get_recommendations_bm25(movie):
    idx = movies[movies['title'] == movie].index[0]
    query = movies['tags'].iloc[idx].split()
    scores = bm25.get_scores(query)
    sim_scores = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[1:6]
    return [movies.iloc[i[0]].title for i in sim_scores]

def get_recommendations_word2vec(movie):
    idx = movies[movies['title'] == movie].index[0]
    sim_scores = sorted(list(enumerate(cosine_sim_word2vec[idx])), key=lambda x: x[1], reverse=True)[1:6]
    return [movies.iloc[i[0]].title for i in sim_scores]

def get_recommendations_jaccard(movie):
    idx = movies[movies['title'] == movie].index[0]
    sim_scores = sorted([(i, jaccard_score(binary_vectors[idx], binary_vectors[i])) for i in range(len(movies))],
                        key=lambda x: x[1], reverse=True)[1:6]
    return [movies.iloc[i[0]].title for i in sim_scores]

# ========== Ensemble Model ==========
def get_ensemble_recommendations(movie):
    recommendations = []
    recommendations += get_recommendations_tfidf(movie)
    recommendations += get_recommendations_lsi(movie)
    recommendations += get_recommendations_bm25(movie)
    recommendations += get_recommendations_word2vec(movie)
    recommendations += get_recommendations_jaccard(movie)

    # Majority Voting
    top_recommendations = Counter(recommendations).most_common(5)
    return [rec[0] for rec in top_recommendations]

# ========== Streamlit UI ==========
st.title('🎬 Movie Recommendation System')
st.subheader("Find similar movies using multiple recommendation models")

# Movie Selection Dropdown
selected_movie = st.selectbox("Choose a movie:", movies['title'].values)

if st.button("Recommend"):
    # Get recommendations from all models
    tfidf_recommendations = get_recommendations_tfidf(selected_movie)
    lsi_recommendations = get_recommendations_lsi(selected_movie)
    bm25_recommendations = get_recommendations_bm25(selected_movie)
    word2vec_recommendations = get_recommendations_word2vec(selected_movie)
    jaccard_recommendations = get_recommendations_jaccard(selected_movie)
    ensemble_recommendations = get_ensemble_recommendations(selected_movie)

    # Display recommendations for each algorithm
    models = {
        "TF-IDF Recommendations": tfidf_recommendations,
        "LSI Recommendations": lsi_recommendations,
        "BM25 Recommendations": bm25_recommendations,
        "Word2Vec Recommendations": word2vec_recommendations,
        "Jaccard Similarity Recommendations": jaccard_recommendations,
        "Ensemble Recommendations": ensemble_recommendations,
    }

    for model_name, recommendations in models.items():
        st.subheader(model_name)
        recommended_posters = [fetch_poster(movies[movies['title'] == movie].iloc[0].movie_id) for movie in recommendations]

        # Display in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(recommendations[0])
            st.image(recommended_posters[0])
        with col2:
            st.text(recommendations[1])
            st.image(recommended_posters[1])
        with col3:
            st.text(recommendations[2])
            st.image(recommended_posters[2])
        with col4:
            st.text(recommendations[3])
            st.image(recommended_posters[3])
        with col5:
            st.text(recommendations[4])
            st.image(recommended_posters[4])

        st.markdown("---")  # Add a separator
