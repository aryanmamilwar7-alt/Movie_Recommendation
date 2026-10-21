# Movie Recommendation System (ML Lab - Group Project)

A recommendation website: the visitor picks favourite genres and rates movies
they have watched, and the page recommends movies they are likely to enjoy.

## Open the website (no installation needed)
Double-click **index.html**. It opens in your browser and works offline.

## How the model works
`movie_recommender.py` trains on the ratings in `data/` and builds a
movie-to-movie similarity matrix from
1. Content-based filtering: TF-IDF on genres + cosine similarity
2. Item-based collaborative filtering: cosine similarity on user-mean-centred ratings

`build_website.py` embeds that matrix into `index.html`. In the page, each
unseen movie is scored by how similar it is to the movies you rated (weighted by
your stars), mixed with how well its genres match your picks (slider) and a small
bonus for movies other viewers rated highly.

## Rebuild the website after changing data or code
    pip install -r requirements.txt
    py build_website.py

## Evaluate the models (MSPA-III / IV)
    py movie_recommender.py
Prints hyperparameter tuning, RMSE / MAE / Precision@10 for popularity, content,
KNN, SVD and hybrid models (see sample_output.txt).

## Data
`data/` holds a generated demo dataset (400 users, 56 movies, ~9k ratings).
For real results, replace `data/movies.csv` and `data/ratings.csv` with MovieLens
`ml-latest-small` (columns: movieId,title,genres / userId,movieId,rating) and
run `py build_website.py` again. The page then uses the 400 most-rated movies.
"# Movie_Recommendation_System" 
