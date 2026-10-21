"""
Movie Recommendation System  (ML Lab - Group Project)
=====================================================
MSPA-I   : Problem statement  -> recommend movies a user is likely to enjoy
MSPA-II  : Dataset + preprocessing (MovieLens files if present, else generated)
MSPA-III : Train and test models (RMSE / MAE on held-out ratings, Precision@K)
MSPA-IV  : Hyperparameter tuning (K for KNN, components for SVD) + final model

Models
  1. Popularity baseline
  2. Content-based filtering  (TF-IDF on genres + cosine similarity)
  3. Item-based Collaborative Filtering (KNN, cosine similarity)
  4. Matrix Factorization (SVD on mean-centred user-item matrix)
  5. Hybrid = average of 2, 3 and 4

Run:
    python movie_recommender.py                  # full pipeline + demo
    python movie_recommender.py --user 5         # recommendations for user 5
    python movie_recommender.py --similar "Inception"

Using real data: replace data/movies.csv (movieId,title,genres) and
data/ratings.csv (userId,movieId,rating) with the MovieLens files
(https://grouplens.org/datasets/movielens/latest/ -> ml-latest-small).
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(42)

# --------------------------------------------------------------------------
# MSPA-II : Dataset
# --------------------------------------------------------------------------
CATALOG = [
    ("The Dark Knight", "Action|Crime|Drama"), ("Inception", "Action|Sci-Fi|Thriller"),
    ("Interstellar", "Adventure|Drama|Sci-Fi"), ("The Matrix", "Action|Sci-Fi"),
    ("Avengers: Endgame", "Action|Adventure|Sci-Fi"), ("Iron Man", "Action|Adventure|Sci-Fi"),
    ("Mad Max: Fury Road", "Action|Adventure|Sci-Fi"), ("John Wick", "Action|Crime|Thriller"),
    ("Die Hard", "Action|Thriller"), ("Gladiator", "Action|Adventure|Drama"),
    ("Titanic", "Drama|Romance"), ("The Notebook", "Drama|Romance"),
    ("La La Land", "Comedy|Drama|Musical|Romance"), ("Pride & Prejudice", "Drama|Romance"),
    ("Casablanca", "Drama|Romance|War"), ("Forrest Gump", "Comedy|Drama|Romance"),
    ("The Hangover", "Comedy"), ("Superbad", "Comedy"), ("Bridesmaids", "Comedy|Romance"),
    ("3 Idiots", "Comedy|Drama"), ("Dumb and Dumber", "Comedy"),
    ("The Conjuring", "Horror|Mystery|Thriller"), ("Get Out", "Horror|Mystery|Thriller"),
    ("A Quiet Place", "Drama|Horror|Sci-Fi"), ("It", "Horror"), ("Hereditary", "Drama|Horror|Mystery"),
    ("Toy Story", "Animation|Children|Comedy"), ("Finding Nemo", "Adventure|Animation|Children"),
    ("The Lion King", "Adventure|Animation|Children|Drama|Musical"), ("Shrek", "Animation|Children|Comedy|Fantasy"),
    ("Frozen", "Animation|Children|Comedy|Musical"), ("Coco", "Adventure|Animation|Children|Fantasy"),
    ("Spirited Away", "Adventure|Animation|Fantasy"),
    ("The Shawshank Redemption", "Crime|Drama"), ("The Godfather", "Crime|Drama"),
    ("Pulp Fiction", "Comedy|Crime|Drama|Thriller"), ("Se7en", "Crime|Mystery|Thriller"),
    ("Gone Girl", "Drama|Mystery|Thriller"), ("Zodiac", "Crime|Drama|Mystery|Thriller"),
    ("Knives Out", "Comedy|Crime|Drama|Mystery"), ("Shutter Island", "Drama|Mystery|Thriller"),
    ("Lord of the Rings: Fellowship", "Adventure|Fantasy"), ("Harry Potter 1", "Adventure|Children|Fantasy"),
    ("The Hobbit", "Adventure|Fantasy"), ("Pan's Labyrinth", "Drama|Fantasy|War"),
    ("Blade Runner 2049", "Drama|Sci-Fi"), ("Arrival", "Drama|Sci-Fi"), ("Ex Machina", "Drama|Sci-Fi|Thriller"),
    ("Saving Private Ryan", "Action|Drama|War"), ("Schindler's List", "Drama|War"),
    ("1917", "Action|Drama|War"), ("Dunkirk", "Action|Drama|Thriller|War"),
    ("Dangal", "Action|Drama"), ("Lagaan", "Adventure|Drama|Musical"),
    ("Dilwale Dulhania Le Jayenge", "Comedy|Drama|Musical|Romance"), ("Zindagi Na Milegi Dobara", "Adventure|Comedy|Drama"),
]
GENRES = sorted({g for _, gs in CATALOG for g in gs.split("|")})


def generate_data(n_users=400):
    """Synthetic but structured ratings: each user has genre tastes."""
    movies = pd.DataFrame({
        "movieId": range(1, len(CATALOG) + 1),
        "title": [t for t, _ in CATALOG],
        "genres": [g for _, g in CATALOG],
    })
    movie_genres = np.array([[g in gs.split("|") for g in GENRES] for _, gs in CATALOG], float)
    quality = RNG.normal(0, 0.4, len(CATALOG))          # some movies are just better
    rows = []
    for u in range(1, n_users + 1):
        taste = RNG.normal(0, 1, len(GENRES))
        taste[RNG.choice(len(GENRES), 3, replace=False)] += 2.0   # 3 favourite genres
        n_rated = RNG.integers(12, 35)
        seen = RNG.choice(len(CATALOG), n_rated, replace=False)
        for m in seen:
            affinity = movie_genres[m] @ taste / max(movie_genres[m].sum(), 1)
            r = 3.2 + 0.6 * affinity + quality[m] + RNG.normal(0, 0.5)
            rows.append((u, m + 1, float(np.clip(np.round(r * 2) / 2, 0.5, 5.0))))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
    return movies, ratings


def load_data():
    # looks in ./data first, then next to the script
    for folder in (os.path.join(HERE, "data"), HERE):
        mp, rp = os.path.join(folder, "movies.csv"), os.path.join(folder, "ratings.csv")
        if os.path.exists(mp) and os.path.exists(rp):
            break
    if os.path.exists(mp) and os.path.exists(rp):
        print(f"[data] Loading movies.csv / ratings.csv from {folder}")
        movies, ratings = pd.read_csv(mp), pd.read_csv(rp)[["userId", "movieId", "rating"]]
    else:
        print("[data] CSV files not found -> using generated dataset")
        movies, ratings = generate_data()
    # preprocessing: keep movies/users with enough ratings, drop dupes/nulls
    ratings = ratings.dropna().drop_duplicates(["userId", "movieId"])
    mc = ratings.movieId.value_counts()
    ratings = ratings[ratings.movieId.isin(mc[mc >= 5].index)]
    uc = ratings.userId.value_counts()
    ratings = ratings[ratings.userId.isin(uc[uc >= 5].index)]
    movies = movies[movies.movieId.isin(ratings.movieId.unique())].reset_index(drop=True)
    movies["genres"] = movies["genres"].fillna("").str.replace("(no genres listed)", "", regex=False)
    return movies, ratings.reset_index(drop=True)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class Recommender:
    def __init__(self, movies, train, k=20, n_components=15):
        self.movies, self.k, self.n_components = movies, k, n_components
        self.uid = {u: i for i, u in enumerate(sorted(train.userId.unique()))}
        self.mid = {m: i for i, m in enumerate(movies.movieId)}
        self.mid_inv = {i: m for m, i in self.mid.items()}
        n_u, n_m = len(self.uid), len(self.mid)

        R = np.zeros((n_u, n_m))
        for u, m, r in train[["userId", "movieId", "rating"]].itertuples(index=False):
            if m in self.mid:
                R[self.uid[u], self.mid[m]] = r
        self.R, self.mask = R, R > 0
        self.global_mean = R[self.mask].mean()
        cnt = self.mask.sum(0)
        self.item_mean = np.where(cnt > 0, R.sum(0) / np.maximum(cnt, 1), self.global_mean)
        ucnt = self.mask.sum(1)
        self.user_mean = np.where(ucnt > 0, R.sum(1) / np.maximum(ucnt, 1), self.global_mean)

        # --- content-based: TF-IDF over genres
        docs = movies["genres"].str.replace("|", " ", regex=False).str.replace("-", "", regex=False)
        self.tfidf = TfidfVectorizer(token_pattern=r"[A-Za-z]+").fit_transform(docs)
        self.content_sim = cosine_similarity(self.tfidf)

        # --- item-based CF: cosine on user-mean-centred ratings
        C = np.where(self.mask, R - self.user_mean[:, None], 0.0)
        self.item_sim = cosine_similarity(C.T)
        np.fill_diagonal(self.item_sim, 0)
        np.fill_diagonal(self.content_sim, 0)

        # --- SVD matrix factorisation on centred matrix
        svd = TruncatedSVD(n_components=min(n_components, n_m - 1), random_state=42)
        U = svd.fit_transform(C)
        self.svd_pred = U @ svd.components_ + self.user_mean[:, None]

    # ---- rating prediction for (user, movie) ----
    def _knn_pred(self, u, m, sim=None):
        sim = self.item_sim if sim is None else sim
        rated = np.where(self.mask[u])[0]
        if len(rated) == 0:
            return self.item_mean[m]
        sims = sim[m, rated]
        top = np.argsort(-np.abs(sims))[: self.k]
        s, idx = sims[top], rated[top]
        if np.abs(s).sum() < 1e-9:
            return self.user_mean[u]
        return self.user_mean[u] + (s * (self.R[u, idx] - self.user_mean[u])).sum() / np.abs(s).sum()

    def predict(self, user, movie, method="hybrid"):
        if user not in self.uid or movie not in self.mid:
            return self.global_mean
        u, m = self.uid[user], self.mid[movie]
        if method == "popularity":
            p = self.item_mean[m]
        elif method == "knn":
            p = self._knn_pred(u, m)
        elif method == "svd":
            p = self.svd_pred[u, m]
        elif method == "content":
            p = self._knn_pred(u, m, self.content_sim)
        else:  # hybrid = average of content-based + item-KNN + SVD
            p = (self._knn_pred(u, m, self.content_sim) + self._knn_pred(u, m)
                 + self.svd_pred[u, m]) / 3
        return float(np.clip(p, 0.5, 5.0))

    # ---- top-N for a user ----
    def recommend(self, user, n=10, method="hybrid"):
        if user not in self.uid:   # cold-start user -> most popular well-rated
            order = np.argsort(-self.item_mean)[:n]
            return [(self.movies.title[i], self.movies.genres[i], round(self.item_mean[i], 2)) for i in order]
        u = self.uid[user]
        unseen = np.where(~self.mask[u])[0]
        scores = [(i, self.predict(user, self.mid_inv[i], method)) for i in unseen]
        scores.sort(key=lambda x: -x[1])
        return [(self.movies.title[i], self.movies.genres[i], round(s, 2)) for i, s in scores[:n]]

    # ---- content-based similar movies ----
    def similar(self, title, n=8):
        hits = self.movies[self.movies.title.str.lower().str.contains(title.lower(), regex=False)]
        if hits.empty:
            return None
        i = hits.index[0]
        blend = 0.5 * self.content_sim[i] + 0.5 * np.clip(self.item_sim[i], 0, None)
        blend[i] = -1
        order = np.argsort(-blend)[:n]
        return self.movies.title[i], [(self.movies.title[j], self.movies.genres[j], round(blend[j], 3)) for j in order]


# --------------------------------------------------------------------------
# MSPA-III : Evaluation
# --------------------------------------------------------------------------
def evaluate(model, test, method):
    preds = np.array([model.predict(u, m, method) for u, m in zip(test.userId, test.movieId)])
    err = preds - test.rating.values
    return float(np.sqrt((err ** 2).mean())), float(np.abs(err).mean())


def precision_at_k(model, test, method, k=10, thresh=4.0):
    """Of the top-k recommended unseen movies, how many did the user later rate >= thresh."""
    hits = tot = 0
    for u, grp in test.groupby("userId"):
        liked = set(grp[grp.rating >= thresh].movieId)
        if not liked or u not in model.uid:
            continue
        cand = grp.movieId.tolist()           # rank only the user's held-out movies
        ranked = sorted(cand, key=lambda m: -model.predict(u, m, method))[:k]
        hits += sum(m in liked for m in ranked)
        tot += min(k, len(cand))
    return hits / tot if tot else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, help="recommend for this userId")
    ap.add_argument("--similar", type=str, help="movies similar to this title")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    movies, ratings = load_data()
    print(f"[data] {ratings.userId.nunique()} users, {len(movies)} movies, {len(ratings)} ratings "
          f"(sparsity {1 - len(ratings) / (ratings.userId.nunique() * len(movies)):.1%})")

    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    val_tr, val = train_test_split(train, test_size=0.2, random_state=1)

    # ---- MSPA-IV: hyperparameter tuning on a validation split
    print("\n=== Hyperparameter tuning (validation RMSE) ===")
    best_k, best_c, bk, bc = None, None, 9, 9
    for k in (5, 10, 20, 40):
        rm, _ = evaluate(Recommender(movies, val_tr, k=k), val, "knn")
        print(f"  KNN  k={k:<3d} RMSE={rm:.4f}")
        if rm < bk: bk, best_k = rm, k
    for c in (5, 10, 15, 25):
        rm, _ = evaluate(Recommender(movies, val_tr, n_components=c), val, "svd")
        print(f"  SVD  components={c:<3d} RMSE={rm:.4f}")
        if rm < bc: bc, best_c = rm, c
    print(f"  -> best k={best_k}, best components={best_c}")

    # ---- final model on full train set, tested on held-out test set
    model = Recommender(movies, train, k=best_k, n_components=best_c)
    print("\n=== Test-set performance ===")
    print(f"{'Model':<12}{'RMSE':>8}{'MAE':>8}{'P@10':>8}")
    for name in ("popularity", "content", "knn", "svd", "hybrid"):
        rm, ma = evaluate(model, test, name)
        p = precision_at_k(model, test, name)
        print(f"{name:<12}{rm:>8.4f}{ma:>8.4f}{p:>8.3f}")

    # ---- refit on ALL ratings so demo recommendations exclude everything the user has rated
    model = Recommender(movies, ratings, k=best_k, n_components=best_c)

    # ---- demo
    user = args.user if args.user else int(ratings.userId.iloc[0])
    print(f"\n=== Top {args.n} recommendations for user {user} (hybrid) ===")
    seen = ratings[ratings.userId == user].merge(movies, on="movieId").sort_values("rating", ascending=False)
    print("Already liked:", ", ".join(seen.title.head(4)))
    for i, (t, g, s) in enumerate(model.recommend(user, args.n), 1):
        print(f"{i:>2}. {t:<32} {g:<38} predicted {s}")

    q = args.similar or movies.title.iloc[1]
    res = model.similar(q)
    print(f"\n=== Movies similar to '{q}' ===")
    if res is None:
        print("  title not found")
    else:
        print("(matched:", res[0] + ")")
        for t, g, s in res[1]:
            print(f"  {t:<32} {g:<38} score {s}")


if __name__ == "__main__":
    main()
