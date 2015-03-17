import MeCab
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

num_clusters = 20

def split_to_words(text):
    m = MeCab.Tagger('-Ochasen')
    res = m.parse(text)
    info_words = res.split('\n')
    words = []
    for info_word in info_words:
        elems = info_word.split('\t')
        if len(elems) != 6: 
            continue 
        words.append(elems[2])
    return words


vectorizer = TfidfVectorizer(analyzer=split_to_words, min_df=1, max_df=50)
files = np.array([ f for f in os.listdir('entries') if f.startswith('entry-') ])
corpus = [ open('entries/' + file, 'r').read() for file in files ]
vectorized = vectorizer.fit_transform(corpus)

num_sample, num_features = vectorized.shape
print("#samples: %d, #features: %d" % (num_sample, num_features))

km = KMeans(n_clusters=num_clusters, init='k-means++', n_init=20, max_iter=10000, verbose=1)
km.fit(vectorized)

result = np.array(km.labels_)
print(result)
for n in range(0, num_clusters): 
    print("Cluster %d ------" % n)
    print(files[result == n])


