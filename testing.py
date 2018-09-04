import numpy as np
from gensim.models import KeyedVectors
from sklearn import linear_model, datasets
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=200000)

# import some data to play with
X = ['happy', "glad", 'smiling', 'super', 'awesome', 'sad', 'evil', 'angry', 'depressing', 'bad']
for i, w in enumerate(X):
    X[i] = model[w]

print(np.array(X).shape)

Y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X, Y)
print(logreg.score(X, Y))
print(logreg.predict_proba(model['happy'].reshape(1, -1))[0][1])
print(logreg.predict_proba(model['sad'].reshape(1, -1))[0][1])
print(logreg.predict_proba(model['depressed'].reshape(1, -1))[0][1])
print(logreg.predict_proba(model['unhappy'].reshape(1, -1))[0][1])
print(logreg.predict_proba(model['enthusiastic'].reshape(1, -1))[0][1])
