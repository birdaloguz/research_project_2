import pandas as pd
import scipy
from scipy import sparse
import numpy as np
from sklearn.model_selection import train_test_split
import pysparnn.cluster_index as ci
from scipy import sparse
coo_row = []
coo_col = []
coo_val = []

datasets = ['netflix/combined_data_1.txt',
            'netflix/combined_data_2.txt']

for dataset in datasets:
    with open(dataset, "r") as f:
        movie = -1
        c=0
        for line in f:
            print(c)
            c+=1
            if line.endswith(':\n'):
                movie = int(line[:-2]) - 1
                continue
            assert movie >= 0
            splitted = line.split(',')
            user = int(splitted[0])
            rating = float(splitted[1])
            coo_row.append(user)
            coo_col.append(movie)
            coo_val.append(rating)

coo_val = np.array(coo_val, dtype=np.float32)
coo_col = np.array(coo_col, dtype=np.int32)
coo_row = np.array(coo_row)
user, indices = np.unique(coo_row, return_inverse=True)
user = user.astype(np.int32)


um_matrix = sparse.coo_matrix((coo_val, (indices, coo_col))).T
#matrix_df = pd.DataFrame(data=um_matrix.toarray(), columns=[i for i in range(81472)])
matrix_df = pd.DataFrame(data=um_matrix.toarray(), columns=[i for i in range(478018)])
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

# ndcg calculation
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

# mrp calculation
def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


eval_results = {'recalls': {},
                'mrr': {},
                'ndcg': {}}

doc_index = np.array(range(len(matrix_df)))

snn = ci.MultiClusterIndex(um_matrix, doc_index, num_indexes=10)
for k in [2, 5, 10, 15, 20, 50, 100]:
    # for each user get average distance of the movies that user rated to retrieve top k movies to recommend
    print(k)

    results = snn.search(um_matrix, k=k, return_distance=True, k_clusters=1)

    results_dic = []
    for i in results:
        results_dic.append(dict((int(y), x) for x, y in i))

    # for each user get average distance of the movies that user rated to retrieve top k movies to recommend
    avg_dict = {}
    for index, user in enumerate(user_hists):
        user_dict = {}
        for movie in user:
            for m in results_dic[movie]:
                if m not in user_dict:
                    user_dict[m] = results_dic[movie][m]
                else:
                    user_dict[m] += results_dic[movie][m]
        avg_dict[index] = dict(sorted(user_dict.items(), key=lambda x: x[1]))
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
    print(recall)

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
    ndcg = 0
    for i in results:
        ndcg += ndcg_at_k(i, 10)
    ndcg = ndcg / len(results)

    eval_results['recalls'][k] = recall
    eval_results['mrr'][k] = mrr
    eval_results['ndcg'][k] = ndcg
    import json

    with open('pysparnn_eval_results_k_netflix.json', 'w') as fp:
        json.dump(eval_results, fp)




