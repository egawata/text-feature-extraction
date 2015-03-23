import MeCab
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import scipy as sp

NUM_CLUSTERS = 20
ENTRIES_TRAINING = 'entries/training/'
ENTRIES_TEST     = 'entries/test/'

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

argvs = sys.argv
if (len(argvs) >= 2):
    file_test = argvs[1]
else:
    files_test = np.array([ f for f in os.listdir(ENTRIES_TEST) if f.startswith('entry-') ])
    file_test = ENTRIES_TEST + files_test[0]

vectorizer = TfidfVectorizer(analyzer=split_to_words, min_df=1, max_df=50, max_features=20000)
files_train = np.array([ f for f in os.listdir(ENTRIES_TRAINING) if f.startswith('entry-') ])
corpus = [ open(ENTRIES_TRAINING + file, 'r').read() for file in files_train ]
vectorized = vectorizer.fit_transform(corpus)

num_sample, num_features = vectorized.shape
print("#samples: %d, #features: %d" % (num_sample, num_features))

km = KMeans(n_clusters=NUM_CLUSTERS, init='k-means++', n_init=10, verbose=1)
km.fit(vectorized)

result = np.array(km.labels_)
print(result)
for n in range(0, NUM_CLUSTERS): 
    print("Cluster %d ------" % n)
    print(files_train[result == n])

#  テストデータを分類する
new_post = open(file_test, 'r').read()
new_post_vec = vectorizer.transform([new_post])
new_post_label = km.predict(new_post_vec)[0]
print("Label of %s is %d" % (file_test, new_post_label))

similar_indices = (km.labels_ == new_post_label).nonzero()[0]
similar = []
for i in similar_indices:
    dist = sp.linalg.norm((new_post_vec - vectorized[i]).toarray())
    similar.append((dist, corpus[i]))

similar = sorted(similar)

print('New Post\n' + '-' * 50 + "\n" + new_post + "\n" +  '-' * 50 + "\n\n")
for sim in similar:
    print(str(sim[0]) + "\n" + '-' * 50 + "\n" + sim[1] + "\n" + '-' * 50 + "\n\n")
    
 
