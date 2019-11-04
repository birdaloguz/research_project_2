import pandas as pd
import scipy
from scipy import sparse
import numpy as np
from sklearn.model_selection import train_test_split

# df_ratings = pd.read_csv('ml-20m/ratings.csv', skiprows=[0], names=["user_id", "movie_id", "rating", "timestamp"]).drop(columns=['timestamp']).head(10000000)
df_ratings = pd.read_csv('ml-10m/ratings.dat', names=["user_id", "movie_id", "rating", "timestamp"],
                         header=None, sep='::', engine='python')
matrix_df = df_ratings.pivot(index='movie_id', columns='user_id', values='rating').fillna(0).astype(bool).astype(int)

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

# list of movies that each user rated
user_hists = []
for user in matrix_df:
    a = [i for i, e in enumerate(matrix_df[user].tolist()) if e != 0]
    user_hists.append(a)

matrix_df = matrix_df.values.tolist()


from annoy import AnnoyIndex
import random

f = len(matrix_df[0])
t = AnnoyIndex(f, "angular")  # Length of item vector that will be indexed
for i in range(0, len(matrix_df)):
    v = matrix_df[i]
    t.add_item(i, v)

t.build(1000)  # 10 trees
t.save('annoy.ann')

t = AnnoyIndex(f, "angular")
t.load('annoy.ann')  # super fast, will just mmap the file

movie_results = []
for i in range(0, len(matrix_df)):
    indices, distances = t.get_nns_by_item(i, 2, include_distances=True)  # will find the all distances
    movie_results.append([indices, distances])

# for each user get average distance of the movies that user rated to retrieve top k movies to recommend
avg_dict = {}
for index, user in enumerate(user_hists):
    user_dict = {}
    for movie in user:
        indices = movie_results[movie][0]
        distances = movie_results[movie][1]
        for idx, i in enumerate(indices):
            if i not in user_dict:
                user_dict[i] = 1 - (((distances[idx] * distances[idx]) - 2) / -2)
            else:
                user_dict[i] += 1 - (((distances[idx] * distances[idx]) - 2) / -2)
    avg_dict[index] = dict(sorted(user_dict.items(), key=lambda x: x[1], reverse=False))
    for m in user_hists[index]:
        try:
            del avg_dict[index][m]
        except:
            pass
    avg_dict[index] = {k: avg_dict[index][k] for k in list(avg_dict[index])[:10]}


recall = 0
for user in avg_dict:
    if validation_movies[user + 1] in avg_dict[user]:
        recall += 1
print(recall)
recall = recall / len(user_hists)


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

results = []
for key in avg_dict:
    results.append(list(avg_dict[key].keys()))


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


for i in range(0, len(results)):
    for x in range(0, len(results[i])):
        if results[i][x] == validation_movies[i + 1]:
            results[i][x] = 1
        else:
            results[i][x] = 0


mrr = mean_reciprocal_rank(results)
ndcg = 0
for i in results:
    ndcg += ndcg_at_k(i, 10)
ndcg = ndcg / len(results)


print(recall)
print(mrr)
print(ndcg)
