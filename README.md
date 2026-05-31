# Anime Recommendation System

A machine learning-powered Anime Recommendation System that combines Content-Based Filtering and Collaborative Filtering into a Hybrid Recommendation Engine. The system recommends anime based on similarities in content, user preferences, and a weighted hybrid approach. Built using Python, Scikit-Learn, Streamlit, and the Jikan API.

---

## Features

### Content-Based Filtering

Recommends anime based on similarities in genres, themes, and descriptions using TF-IDF vectorization and cosine similarity.

### Collaborative Filtering

Uses K-Nearest Neighbors (KNN) on user-anime rating data to identify anime preferred by users with similar viewing patterns.

### Hybrid Recommendation System

Combines content-based and collaborative recommendations using weighted scoring to provide more balanced and accurate recommendations.

### Fuzzy Search

Handles spelling mistakes and partial anime names using RapidFuzz.

### Anime Metadata Integration

Fetches anime posters, ratings, episode counts, and genres using the Jikan API.

### Interactive Web Application

Built with Streamlit for an easy-to-use and responsive interface.

---

## Project Architecture

```text
User Input
     │
     ▼
 Fuzzy Search
     │
     ▼
─────────────────────────────────────
│                                   │
▼                                   ▼
Content-Based Model          Collaborative Model
(TF-IDF + Cosine)            (KNN + Sparse Matrix)
│                                   │
▼                                   ▼
Content Recommendations     User-Based Recommendations
│                                   │
└──────────────┬────────────────────┘
               ▼
        Hybrid Recommender
               ▼
         Final Results
               ▼
       Streamlit Interface
```

---

## Machine Learning Techniques Used

### Content-Based Filtering

* TF-IDF Vectorization
* Cosine Similarity
* Text Feature Engineering

### Collaborative Filtering

* Sparse Matrix Representation
* K-Nearest Neighbors (KNN)
* Cosine Distance

### Hybrid Recommendation

* Weighted score combination
* Recommendation merging
* Ranking and sorting

---

## Dataset

The project uses two anime datasets:

### Anime Information Dataset

Contains:

* Anime ID
* Anime Name
* Genres
* Synopsis
* Type
* Episodes
* Rating

### User Rating Dataset

Contains:

* User ID
* Anime ID
* User Rating

These datasets are cleaned and processed before model training.

---

## Tech Stack

### Programming Language

* Python

### Machine Learning

* Scikit-Learn
* NumPy
* Pandas
* SciPy

### Recommendation Algorithms

* TF-IDF
* Cosine Similarity
* K-Nearest Neighbors (KNN)

### Web Application

* Streamlit

### API

* Jikan Anime API

### Model Serialization

* Joblib

---

## Project Structure

```text
PROJECT1/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── knn_model.pkl
│   ├── sparse_matrix.npz
│   ├── ani_to_index.pkl
│   ├── index_to_ani.pkl
│   ├── tf_idf.pkl
│   ├── tf_idf_matrix.pkl
│   ├── top_sim.pkl
│   ├── ind.pkl
│   └── content_anime_index.pkl
│
├── notebooks/
│   ├── cleaning.ipynb
│   ├── collab_filtering.ipynb
│   ├── content_based_recommendation.ipynb
│   └── hybrid.ipynb
│
├── requirements.txt
│
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/dattasai6788/anime-recommendation-system.git
cd anime-recommendation-system
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app/streamlit_app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## Challenges Faced During Development

* Handling large similarity matrices causing memory issues.
* Sparse matrix optimization for collaborative filtering.
* Maintaining consistency between serialized model artifacts.
* Resolving casing inconsistencies between datasets.
* Integrating multiple recommendation models into a hybrid system.
* Debugging notebook-to-Streamlit deployment differences.

---

## Future Improvements

* Matrix Factorization (SVD)
* Netflix-style recommendation UI
* User accounts
* Watchlist functionality
* Personalized recommendations
* Recommendation explanations
* Anime trailer integration
* Advanced ranking models
* Fast similarity search using FAISS

---

## Learning Outcomes

Through this project I gained practical experience in:

* Recommendation Systems
* Machine Learning Pipelines
* Data Cleaning
* Feature Engineering
* Sparse Matrix Operations
* Model Serialization
* API Integration
* Streamlit Deployment
* Debugging Production Issues
* Hybrid Recommendation Architectures

---

### Outputs

<img width="1894" height="978" alt="Screenshot 2026-05-31 195220" src="https://github.com/user-attachments/assets/3270af48-e34e-470e-9ec2-775975e8a746" />
<img width="1900" height="981" alt="Screenshot 2026-05-31 195253" src="https://github.com/user-attachments/assets/c490a23b-8937-413d-8ffb-d7a837502e58" />

