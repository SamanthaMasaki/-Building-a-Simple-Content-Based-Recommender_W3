# DSA 4060 – Week 3 Practical Lab
# Content-Based Movie Recommendation System

## 1. Project Overview

This project develops a content-based movie recommendation system using Python and Pandas. The system recommends movies to users based on their ratings and the genre characteristics of movies they have previously watched.

The model represents each movie using five genre features: Action, Comedy, Drama, Romance, and SciFi. It uses these features and the user's ratings to build a user preference profile, calculate recommendation scores, rank movies, and recommend movies the user has not watched.

This practical demonstrates the fundamental principles of content-based filtering and how item metadata can be used to generate personalized recommendations.

## 2. Project Objectives

The main objectives of this practical are to:

- Explore and understand a structured movie dataset.
- Represent movies using binary genre features.
- Create user ratings and connect them to movie metadata.
- Construct a weighted user preference profile.
- Normalize the user profile to obtain proportional genre preferences.
- Calculate recommendation scores for movies.
- Rank movies according to their recommendation scores.
- Remove movies the user has already watched.
- Generate Top-N movie recommendations.
- Compare recommendations for different users.
- Add a unique movie to the dataset and evaluate its recommendations.
- Create a personal user profile and interpret its results.
- Identify the limitations of content-based recommendation systems.

## 3. Technologies and Tools

The project uses the following technologies:

| Technology | Purpose |
|---|---|
| Python | Programming language used to implement the recommendation system |
| Pandas | Creating, loading, cleaning, and manipulating tabular data |
| Jupyter Notebook | Writing and executing code alongside explanations and results |
| Visual Studio Code | Development environment used to create and run the notebook |
| Git | Tracking changes to project files |
| GitHub | Hosting and submitting the completed project |

## 4. Project Files

The repository contains the following files:

```text
DSA4060-Week3-StudentID/
│
├── content_based_lab.ipynb
├── movies.csv
└── README.md
```

### File descriptions

**`content_based_lab.ipynb`**

The main Jupyter notebook. It contains the Python code, explanations, calculations, recommendation results, interpretation questions, limitations, and conclusion.

**`movies.csv`**

The dataset containing movie IDs, movie titles, and genre features used by the recommendation system.

**`README.md`**

This document describes the project, its methodology, the required technologies, and the steps needed to run the notebook.

## 5. Dataset Description

The movie dataset contains the following columns:

| Column | Description |
|---|---|
| `movie_id` | Unique identifier assigned to each movie |
| `title` | Name of the movie |
| `Action` | Indicates whether the movie has the Action genre |
| `Comedy` | Indicates whether the movie has the Comedy genre |
| `Drama` | Indicates whether the movie has the Drama genre |
| `Romance` | Indicates whether the movie has the Romance genre |
| `SciFi` | Indicates whether the movie has the Science Fiction genre |

The five genre features are binary variables:

- `1` indicates that the movie contains the genre.
- `0` indicates that the movie does not contain the genre.

The practical uses example movie records to demonstrate the recommendation process. These records can be replaced with a lecturer-provided dataset or an expanded dataset containing additional movies.

## 6. Methodology

The system follows a content-based filtering approach. It compares movie features with a preference profile derived from a user's ratings.

### Step 1: Import libraries

Pandas is imported to support data creation, manipulation, calculations, and CSV operations.

```python
import pandas as pd
```

### Step 2: Create or load the dataset

Movie information is stored in a Pandas DataFrame or loaded from `movies.csv`.

```python
movies = pd.read_csv("movies.csv")
```

If the dataset is created directly in the notebook, it can be saved using:

```python
movies.to_csv("movies.csv", index=False)
```

### Step 3: Explore the dataset

The dataset is inspected to understand its dimensions, columns, data types, and missing values.

```python
movies.head()
movies.shape
movies.info()
movies.isnull().sum()
```

### Step 4: Identify movie features

The five genre columns are selected as the features used by the recommendation algorithm.

```python
features = [
    "Action",
    "Comedy",
    "Drama",
    "Romance",
    "SciFi"
]

movie_features = movies[features]
```

### Step 5: Create user ratings

A ratings DataFrame records the movies watched by a user and the corresponding ratings.

For example:

```python
ratings = pd.DataFrame({
    "movie_id": [1, 3, 6, 9],
    "rating": [5, 4, 2, 3]
})
```

The example represents four movie ratings on a five-star scale.

### Step 6: Construct the weighted user profile

The user's ratings are multiplied by the genre features of the corresponding movies.

```python
user_movies = ratings.merge(
    movies,
    on="movie_id"
)

rated_features = user_movies[features]

weighted_features = rated_features.mul(
    user_movies["rating"],
    axis=0
)

user_profile = weighted_features.sum()
```

The resulting profile summarizes the user's accumulated genre scores. Genres associated with highly rated movies receive larger contributions.

### Step 7: Normalize the user profile

The profile scores are divided by their total to obtain proportions.

```python
user_profile_normalized = (
    user_profile / user_profile.sum()
)
```

Normalization makes the genre preferences easier to interpret and ensures that the normalized scores sum to approximately 1, provided the original profile has a positive total.

### Step 8: Calculate recommendation scores

Each movie receives a score based on the weighted sum of its genre features and the user's normalized preferences.

```python
movies["recommendation_score"] = (
    movies[features]
    .mul(user_profile_normalized)
    .sum(axis=1)
)
```

Movies with genre combinations that align more closely with the user's preferences generally receive higher scores.

### Step 9: Rank movies

The movies are sorted by recommendation score in descending order.

```python
ranked_movies = movies.sort_values(
    "recommendation_score",
    ascending=False
)
```

The highest-scoring movies appear first.

### Step 10: Remove watched movies

Movies whose IDs appear in the user's ratings are excluded from the recommendation results.

```python
watched_movies = ratings["movie_id"]

recommendations = ranked_movies[
    ~ranked_movies["movie_id"].isin(watched_movies)
]
```

This ensures that the system focuses on suggesting unseen movies.

### Step 11: Generate Top-N recommendations

The highest-ranked unseen movies are selected.

```python
top_3 = recommendations[
    ["title", "recommendation_score"]
].head(3)

top_3
```

The value of `n` can be changed to generate a different number of recommendations.

### Step 12: Build a reusable function

A reusable `recommend_movies()` function combines the main steps into a single operation. It allows the system to generate recommendations for different users without manually repeating the calculations.

The function accepts:

- `movies`: The movie dataset.
- `ratings`: The ratings for the selected user.
- `features`: The movie genre columns.
- `n`: The number of recommendations to return.

### Step 13: Compare users

A second user's ratings are provided to determine how different ratings affect the preference profile and the resulting recommendations.

### Step 14: Add a unique movie

A new movie with a unique ID and a selected genre combination is added to the dataset. Its recommendation score is calculated using the same feature-based method.

### Step 15: Create and evaluate a personal user

A personal ratings DataFrame is created using at least four movies. The system generates up to five recommendations, and the resulting preference profile and rankings are interpreted.

## 7. Content-Based Filtering Workflow

The system follows this sequence:

```text
Movie Metadata
      |
      v
Movie Feature Profiles
      |
      v
User Ratings
      |
      v
Weighted User Profile
      |
      v
Normalized Preferences
      |
      v
Recommendation Scores
      |
      v
Remove Watched Movies
      |
      v
Rank Remaining Movies
      |
      v
Top-N Recommendations
```

This workflow shows how structured movie metadata and user ratings are converted into personalized recommendations.

## 8. Interpretation and Evaluation

The notebook evaluates the recommendation system using the following questions:

1. Which two genres have the strongest scores in the user's preference profile?
2. Which movies receive the highest recommendation scores?
3. Why does the highest-ranked movie receive its score?
4. Do different users receive the same recommendations?
5. What can the system learn from a user who has rated only one movie?
6. How does adding a new movie affect the recommendation results?
7. What important information does the model ignore?
8. Should two users who watched the same movies but assigned different ratings receive identical recommendations?

The results are interpreted using the actual profile scores, recommendation scores, movie titles, and ratings produced by the notebook.

Different ratings can change a user's preference profile and may lead to different recommendations. However, users may still share some recommendations when their preferences overlap.

## 9. Limitations

The system has several limitations:

### 9.1 Limited genre features

The model uses only five genre features. It does not consider actors, directors, release dates, languages, plot descriptions, or other movie attributes.

### 9.2 Limited user history

A user with only one or two ratings provides little information about their interests. This can result in recommendations that are too narrow or inaccurate.

### 9.3 Lack of diversity

The model may repeatedly recommend movies with similar genres, limiting the variety of recommendations.

### 9.4 Dependence on metadata quality

Incorrect or incomplete genre labels can affect the accuracy of the calculated recommendation scores.

### 9.5 Simplified user preferences

The model assumes that movie ratings can be used directly as preference weights. It does not distinguish between explicit dislike, neutral preferences, and differences in individual rating behaviour in a more advanced way.

### 9.6 Limited evaluation

The practical primarily demonstrates how recommendations are calculated. It does not establish real-world recommendation quality through a comprehensive user study or a held-out test dataset.

## 10. Possible Improvements

Future versions could improve the system by:

- Adding more movie features, such as actors, directors, release year, language, and plot descriptions.
- Using a larger and more diverse movie dataset.
- Combining content-based filtering with collaborative filtering to create a hybrid recommender.
- Introducing similarity measures such as cosine similarity.
- Incorporating user feedback to update preferences over time.
- Evaluating recommendation quality using suitable offline metrics and held-out user interactions.
- Adding diversity and novelty considerations to the ranking process.

## 11. How to Run the Project

### Prerequisites

Ensure that Python and Visual Studio Code are installed. Install the Python and Jupyter extensions in VS Code.

### Installation

Open the project folder in VS Code and open a terminal.

Install the required packages:

```bash
python -m pip install pandas jupyter ipykernel
```

If your system uses the `py` launcher instead, run:

```bash
py -m pip install pandas jupyter ipykernel
```

### Execution

1. Open `content_based_lab.ipynb` in VS Code.
2. Select a Python kernel.
3. Ensure that `movies.csv` is available in the notebook's working directory if the notebook loads the dataset from that file.
4. Run the notebook cells from top to bottom using `Shift + Enter`.
5. Inspect the generated user profiles, recommendation scores, ranked movies, and Top-N recommendations.
6. Review the interpretation questions and limitations.
7. Restart the kernel and run all cells to verify that the notebook executes from beginning to end.

**Note:** The dataset creation cell and the CSV loading cell are alternatives. Use the appropriate one for your project so that the dataset is not unintentionally replaced or duplicated.

## 12. Expected Outputs

When the notebook runs successfully, it should produce:

- A preview and structural summary of the movie dataset.
- A table of movie genre features.
- User ratings linked to movie metadata.
- Weighted and normalized user preference profiles.
- Recommendation scores for the available movies.
- Ranked movie lists.
- Recommendations that exclude watched movies.
- Top 3 recommendations for the initial user.
- Results for a second user.
- Recommendations after adding a unique movie.
- Top 5 recommendations for a personal user.
- Written interpretations and identified model limitations.

The exact movie rankings depend on the dataset, genre combinations, and ratings used.

## 13. Conclusion

This practical implements a content-based movie recommendation system using Python and Pandas. It demonstrates how movie genre features and user ratings can be used to construct preference profiles and calculate personalized recommendation scores.

By ranking unseen movies and comparing multiple users, the project illustrates how content-based filtering can support personalized recommendations. The practical also highlights the importance of sufficient user history, accurate metadata, feature selection, and recommendation diversity.

Although this implementation is a simplified educational model, it provides a foundation for developing more advanced recommendation systems.
