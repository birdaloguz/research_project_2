import pandas as pd
import numpy as np
import scipy
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
import time

start = time.time()
file1 = open("sknn_results.txt","a")
file1.write("1\n")
file1.close()
df = pd.read_csv('rsc15-raw/rsc15-clicks.dat', names=["session_id", "timestamp", "item_id", "category"],
                         header=None, sep=',', engine='python').drop_duplicates(subset=['session_id', 'item_id'], keep='last')
df['value']=1
df = df.reset_index()

session_validation_dict = {}
drop_index = []

for i in df.session_id.unique()[:200000]:
    d=df.loc[df['session_id'] == i]
    if len(d)>=5:
        session_validation_dict[i] = d.iloc[[1]]['item_id'].values[0]
        drop_index.append(d.iloc[[1]]['item_id'].index)

df.drop(df.index[drop_index], inplace=True)

file1 = open("sknn_results.txt","a")
file1.write("valid\n")
file1.close()

um_matrix = scipy.sparse.csr_matrix((df.value, (df.session_id, df.item_id)))


model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=500, n_jobs=-1)
model_knn.fit(um_matrix)
distances, indices = model_knn.kneighbors(um_matrix)
session_dict = {}

file1 = open("sknn_results.txt","a")
file1.write("training\n")
file1.close()

def combineAll(input):
    result = set(input[0])

    for array in input[1:]:
        result.update(array)

    return list(result)

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


def session_d(i):
    session_dict[i]={}
    session_items = list(df.loc[df['session_id']==i]['item_id'])
    N_s = indices[i]
    N_s_distances = distances[i]
    r_items = [list(df.loc[df['session_id']==s]['item_id']) for s in N_s]
    r_items_combined = combineAll(r_items)
    for item in session_items:
        r_items_combined.remove(item)
    for item in r_items_combined:
        for idx, s in enumerate(N_s):
            if item in r_items[idx]:
                try:
                    session_dict[i][item] += 1 - N_s_distances[idx]
                except:
                    session_dict[i][item] = 1 - N_s_distances[idx]

        session_dict[i]=dict(sorted(session_dict[i].items(), key=lambda x: x[1], reverse=True))

        session_dict[i] = {m: session_dict[i][m] for m in list(session_dict[i])[:10]}
    return session_dict


result = Parallel(n_jobs=-1)(delayed(session_d)(i) for i in session_validation_dict.keys())

file1 = open("sknn_results.txt","a")
file1.write("sessions\n")
file1.close()

session_dict = {}
for d in result:
    session_dict.update(d)


recall = 0
for session in session_validation_dict:
    if session_validation_dict[session] in session_dict[session].keys():
        recall += 1

recall = recall / len(session_validation_dict)
results = []
for session in session_validation_dict:
    x=[]
    for item in session_dict[session]:
        if session_validation_dict[session]==item:
            x.append(1)
        else:
            x.append(0)
    results.append(x)

file1 = open("sknn_results.txt","a")
file1.write("results\n")
file1.close()

mrr = mean_reciprocal_rank(results)

ndcg = 0
for i in results:
    ndcg += ndcg_at_k(i, 10)
ndcg = ndcg / len(results)

stop = time.time()

file1 = open("sknn_results.txt","a")
file1.write(str(recall)+'\n')
file1.write(str(mrr)+'\n')
file1.write(str(ndcg)+'\n')
file1.write('Elapsed time for the entire processing: {:.2f} s'.format(stop - start)+'\n')
file1.close()