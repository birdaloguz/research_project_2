import pandas as pd
import scipy
from scipy import sparse
import numpy as np
from sklearn.model_selection import train_test_split

#df_ratings = pd.read_csv('ml-20m/ratings.csv', skiprows=[0], names=["user_id", "movie_id", "rating", "timestamp"]).drop(columns=['timestamp'])
df_ratings = pd.read_csv('movie_tweetings/ratings.dat', names=["user_id", "movie_id", "rating", "timestamp"],
                         header=None, sep='::', engine='python')
matrix_df = df_ratings.pivot(index='movie_id', columns='user_id', values='rating').fillna(0).astype(bool).astype(int)
print(matrix_df)
# idx to id and reverse dicts
c = 0
hashmap = {}
reverse_hashmap = {}
for i in matrix_df.index.tolist():
    hashmap[c] = i
    reverse_hashmap[i] = c
    c += 1

# get first nonzero elements for validation
validation_movies = matrix_df.ne(0).idxmax()

import random

for col in matrix_df:
    validation_movies[col] = random.choice(matrix_df[col].to_numpy().nonzero()[0])

# make validation movies unrated
for index, row in validation_movies.items():
    matrix_df[index][hashmap[row]] = 0

um_matrix = scipy.sparse.csr_matrix(matrix_df.values)


# idx to id and reverse dicts
c = 0
user_hashmap = {}
user_reverse_hashmap = {}
for i in list(matrix_df):
    user_hashmap[c] = i
    user_reverse_hashmap[i] = c
    c += 1

# list of movies that each user rated
user_hists = []
for user in matrix_df:
    a = [i for i, e in enumerate(matrix_df[user].tolist()) if e != 0]
    user_hists.append(a)


from sklearn.neighbors import NearestNeighbors

# knn model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=2, n_jobs=-1)
model_knn.fit(um_matrix)


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


eval_results = {'recalls': {},
                'mrr': {},
                'ndcg': {}}

for k in [2, 5, 10, 15, 20, 50]:
    print(k)


    distances, indices = model_knn.kneighbors(um_matrix, n_neighbors=k)

    # for each user get average distance of the movies that user rated to retrieve top k movies to recommend
    avg_dict = {}
    for index, user in enumerate(user_hists):
        user_dict = {}
        for movie in user:
            distances_user = distances[movie].squeeze().tolist()
            indices_user = indices[movie].squeeze().tolist()
            for idx, i in enumerate(indices_user):
                if i not in user_dict:
                    user_dict[i] = distances_user[idx]
                else:
                    user_dict[i] += distances_user[idx]
        avg_dict[index] = dict(sorted(user_dict.items(), key=lambda x: x[1], reverse=False))
        for m in user_hists[index]:
            try:
                del avg_dict[index][m]
            except:
                pass
        avg_dict[index] = {k: avg_dict[index][k] for k in list(avg_dict[index])[:10]}

    recall = 0
    for user in avg_dict:
        if validation_movies[user_hashmap[user]] in avg_dict[user]:
            recall += 1

    recall = recall / len(user_hists)
    eval_results['recalls'][k] = recall

    results = []
    for key in avg_dict:
        results.append(list(avg_dict[key].keys()))

    for i in range(0, len(results)):
        for x in range(0, len(results[i])):
            if results[i][x] == validation_movies[user_hashmap[i]]:
                results[i][x] = 1
            else:
                results[i][x] = 0

    mrr = mean_reciprocal_rank(results)
    eval_results['mrr'][k]=mrr
    ndcg = 0
    for i in results:
        ndcg += ndcg_at_k(i, 10)
    ndcg = ndcg / len(results)
    eval_results['ndcg'][k]=ndcg


    import json

    with open('knn_eval_results4_k.json', 'w') as fp:
        json.dump(eval_results, fp)



