# Movie Recommender System Pipeline

## Project Overview
This project implements a comprehensive movie recommendation system pipeline designed for the Department of Computational Linguistics at the University of Tübingen. The system experiments with three major filtering approaches to address information overload and improve user personalization: Content-Based Filtering, Collaborative Filtering, and a Hybrid Model.

Additionally, the project explores advanced techniques, including **Data Augmentation using large language models (LLMs)** and **Transformer-based semantic search**.

## Datasets
The system utilizes two textual datasets sourced from Kaggle:
1.  **MovieLens-latest-small**: Contains approximately 100,000 ratings and serves as the backbone for user-based collaborative filtering.
2.  **TMDb-5000**: A comprehensive dataset providing rich metadata (plot, crew, cast, genres) and weighted average ratings, used for content-based filtering and popularity ranking.

## Features & Methodology

The pipeline implements the following filtering approaches:

### 1. Demographic-Based Filtering
* **Logic:** Offers generalized recommendations based on movie popularity.
* **Implementation:** Calculates a weighted rating score (IMDb formula) to rank movies, prioritizing those with high ratings.

### 2. Content-Based Filtering
* **Logic:** Suggests items similar to those a user liked, based on item attributes.
* **Techniques:**
    * **Textual Similarity:** Uses TF-IDF vectorization on movie overviews with Cosine Similarity.
    * **Metadata Similarity:** Uses CountVectorizer on combined "soup" data (Keywords + Cast + Director + Genres) to compute similarity based on shared entities.
    * **Metrics:** Experiments with Cosine Similarity, Euclidean Distance, and Jaccard Similarity (for genres).

### 3. Collaborative Filtering
* **Logic:** Leverages user behavior and rating patterns to find similar users or items.
* **Techniques:**
    * **Item-Based:** Uses Pearson Correlation to find similarities in rating patterns.
    * **Latent Factor Model:** Implements Singular Value Decomposition (SVD) via the `scikit-surprise` library to predict user ratings, achieving a mean RMSE of approx. 0.87.

### 4. Hybrid Recommender
* **Logic:** Combines Content-Based and Collaborative approaches to overcome limitations like the "cold-start" problem and data sparsity.
* **Implementation:** First retrieves the top 25 movies based on content similarity (TF-IDF/Overview), then re-ranks them using the SVD predicted ratings for the specific user.

### 5. Transformer-Based Semantic Search
* **Logic:** Uses deep learning to capture semantic meaning for natural language queries.
* **Implementation:** Encodes movie metadata using **Sentence-BERT (all-MiniLM-L6-v2)** to allow users to search via free-text queries like "Romantic comedies released in the 1990s".

## Data Augmentation with LLMs
To address missing data fields that could degrade content-based performance, the project employs an instruction-tuned Large Language Model.
* **Problem:** Significant missing values in the `tagline` field for TMDb movies.
* **Solution:** Used **Llama-3.2-3B-Instruct** to generate plausible taglines based on the movie title and overview.
* **Outcome:** Successfully populated missing fields with contextually appropriate text, enhancing the feature set for vectorization.

## Tech Stack
* **Language:** Python
* **Environment:** Jupyter Notebook
* **Libraries:**
    * `pandas`, `numpy` (Data Manipulation)
    * `scikit-learn` (TF-IDF, CountVectorizer, Similarity Metrics)
    * `scikit-surprise` (SVD, Cross-validation)
    * `sentence-transformers`, `torch` (Deep Learning/Embeddings)
    * `kagglehub` (Data Acquisition)

## Qualitative Evaluation

While the SVD model was evaluated quantitatively (RMSE ≈ 0.87), the Hybrid model relies on qualitative inspection to verify that it strikes a balance between relevance and personalization.

**Example Case: *Dr. No***
* **Input:** *Dr. No* (a James Bond film).
* **Hybrid Result:** The system prioritized sequels like *From Russia with Love*.
* **Analysis:** This demonstrates the Hybrid model's strength. A pure Collaborative Filtering model might miss this connection if the user hasn't explicitly rated early Bond films (Cold Start), while a pure Content-Based model might suggest generic action movies. The Hybrid approach successfully surfaced high-quality, franchise-relevant titles.

## Future Work
* **Evaluation Baseline:** Implementation of A/B testing or more rigorous hold-out validation for all models.
* **Multimodal Data:** Integrating movie posters/images for multimodal embeddings.
* **Advanced Models:** Exploration of Neural Collaborative Filtering (NCF) or Knowledge Graphs.

## Author
**Özlem Karabulut**
Department of Computational Linguistics, University of Tübingen
