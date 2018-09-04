from blaze.expr.datetime import is_month_end
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing import preprocessing

from sklearn import linear_model, datasets
import numpy as np

class AbstractNode:
    name = None
    word2vec = None
    logreg = None
    training_uptodate = True
    pos_words = []
    neg_words = []

    def __init__(self, word2vec, name):
        self.name = name
        self.word2vec = word2vec
        self.logreg = linear_model.LogisticRegression(C=1e5)

    def give_pos(self, ew):
        self.pos_words.append(ew)
        self.training_uptodate = False

    def give_neg(self, ew):
        self.neg_words.append(ew)
        self.training_uptodate = False

    def give_word(self, word, is_positive):
        ew = self.get_vector(word)
        if ew is not None:
            if is_positive:
                self.give_pos(ew)
            else:
                self.give_neg(ew)
        else:
            print("  WARNING", word, "not in w2v model")

    def give_desc(self, desc, is_positive):
        desc = self._clean_text(desc)
        if is_positive:
            for w in desc.split(" "): self.give_word(w, True)
        else:
            for w in desc.split(" "): self.give_word(w, False)

    def get_data_shape(self):
        return np.array(self.pos_words).shape, np.array(self.neg_words).shape

    def _helper_get_examples(self, ews):
        words = []
        for ew in ews:
            words.append(self.word2vec.most_similar(ew, topn=1)[0][0])
        return words

    def get_examples(self, is_positive):
        if is_positive:
            return self._helper_get_examples(self.pos_words)
        else:
            return self._helper_get_examples(self.neg_words)

    def get_vector(self, word):
        if word in self.word2vec.vocab:
            return self.word2vec[word].reshape(1, -1)
        else:
            return None

    @staticmethod
    def _clean_text(unclean):
        filters = [lambda x: x.lower(), preprocessing.strip_tags, preprocessing.strip_punctuation,
                   preprocessing.strip_multiple_whitespaces, preprocessing.strip_numeric,
                   preprocessing.remove_stopwords, preprocessing.strip_short]
        result = preprocess_string(unclean, filters=filters)
        return " ".join(result)

    def _helper_get_opposite_vectors(self, vectors):
        num = len(vectors)
        num = max(num, 2)
        same_words = self._helper_get_examples(vectors)
        # opp_words = self.word2vec.most_similar(words, topn=distance)[distance - num:]
        opp_words = self.word2vec.most_similar(negative=same_words, topn=num*100)
        print(opp_words)
        return [self.get_vector(i[0]) for i in opp_words]

    def train(self):
        x = []
        y = []
        if len(self.pos_words) > 0 and len(self.neg_words) > 0:
            x = np.concatenate((self.pos_words, self.neg_words))
            y = np.array(([1] * len(self.pos_words)) + [0] * len(self.neg_words))
        elif len(self.pos_words) > 0:
            opp_words = self._helper_get_opposite_vectors(self.pos_words)
            x = np.concatenate((self.pos_words, opp_words))
            y = np.array(([1] * len(self.pos_words)) + [0] * len(opp_words))
        elif len(self.neg_words) > 0:
            opp_words = self._helper_get_opposite_vectors(self.neg_words)
            x = np.concatenate((self.neg_words, opp_words))
            y = np.array(([1] * len(self.neg_words)) + [0] * len(opp_words))
        else:
            self.logreg = None
            self.training_uptodate = True
            return

        x = x.reshape(x.shape[0], x.shape[2])
        self.logreg = self.logreg.fit(x, y)
        self.training_uptodate = True

    def predict(self, ew):
        if not self.training_uptodate:
            self.train()
        if self.logreg is None:
            return 1.0
        else:
            return self.logreg.predict_proba(ew.reshape(1, -1))[0][1]

    def predict_word(self, word):
        ew = self.get_vector(word)
        return self.predict(ew)

    def predict_desc(self, desc):
        predictions = []
        desc = self._clean_text(desc)
        for w in desc.split(' '):
            predictions.append(self.predict_word(w))
            # print(w, self.predict_word(w))
        return np.percentile(np.array(predictions), 90)

    def ping(self, vector, threshold=0.5):
        p = self.predict(vector)
        if p > threshold:
            return self.name, p

if __name__ == "__main__":
    from gensim.models import KeyedVectors
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=200000)

    print("START")
    a = AbstractNode(model, 'happy')
    a.give_desc("delighted, pleased, or glad, as over a particular thing:", True)
    a.give_desc("feeling or showing sorrow; unhappy.", False)
    print(a.predict_desc('Life is like a roller coaster, live it, be happy, enjoy life. '))
    words = ['sad', 'happy', 'cat', 'feeling']
    for w in words:
        print(w, a.predict_word(w))

