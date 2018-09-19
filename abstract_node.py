from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing import preprocessing
from sklearn import linear_model
import numpy as np
import pandas as pd


class AbstractNode:

    def __init__(self, word2vec, name):
        self.name = name
        self.word2vec = word2vec
        self.logreg_attr = linear_model.LogisticRegression(C=1e5)
        self.logreg_ex = linear_model.LogisticRegression(C=1e5)
        self.logreg_attr_trained = False
        self.logreg_ex_trained = False
        self.name_vector = self.get_vector(name)
        self.data_attr = pd.DataFrame(columns=['x', 'y', 'word'])
        self.data_ex = pd.DataFrame(columns=['x', 'y', 'word'])

    def add_data(self, relation, x, y, is_word=True):
        """
        Adds a pos/neg attribute or example of the node to the data
        :param relation: The kind of relationship to add. Either 'attr' or 'ex'
        :param x: The vector of the word to be added
        :param y: 1 or 0 if the example is pos or neg
        :param is_word: if x needs to be converted to a vector
        """
        # convert to vector
        word = None
        if is_word:
            word = x
            x = self.get_vector(x)
            if x is None:
                print("WARNING COULD NOT ADD", x)
                return

        if relation == 'attr':
            self.data_attr = self.data_attr.append({'x': x, 'y': y * 1, 'word': word}, ignore_index=True)
            self.logreg_attr_trained = False
        elif relation == 'ex':
            self.data_ex = self.data_ex.append({'x': x, 'y': y * 1, 'word': word}, ignore_index=True)
            self.logreg_ex_trained = False
        else:
            print("WARNING INCORRECT TYPE. IGNORING")

    def add_desc(self, relation, desc, is_positive):
        if desc is None:
            return
        for word in self._clean_text(desc).split(" "):
            self.add_data(relation, word, is_positive)

    def get_vector(self, word):
        if word in self.word2vec.vocab:
            return self.word2vec[word].reshape(1, -1)
        else:
            print("WARNING", word, "not in w2v vocab")
            return None

    @staticmethod
    def _clean_text(unclean):
        filters = [lambda x: x.lower(), preprocessing.strip_tags, preprocessing.strip_punctuation,
                   preprocessing.strip_multiple_whitespaces, preprocessing.strip_numeric,
                   preprocessing.remove_stopwords, preprocessing.strip_short]
        result = preprocess_string(unclean, filters=filters)
        return " ".join(result)

    def _helper_get_opposite_words(self, words):
        num = len(words)
        num = max(num, 2)
        # opp_words = self.word2vec.most_similar(words, topn=distance)[distance - num:]
        opp_words = self.word2vec.most_similar(negative=words, topn=num*100)
        # print(opp_words)
        return [i[0] for i in opp_words[:num]]

    def train(self):
        y_attr = np.array(self.data_attr['y'].values, dtype=bool)
        y_ex = np.array(self.data_ex['y'].values, dtype=bool)

        if len(y_attr) > 0:
            if False not in y_attr:
                print("WARNING GETTING OPP WORDS")
                for x in self._helper_get_opposite_words(self.data_attr['word'].values):
                    self.add_data('attr', x, False)
                y_attr = np.array(self.data_attr['y'].values, dtype=bool)

            x_attr = np.concatenate(self.data_attr['x'].values)
            self.logreg_attr = self.logreg_attr.fit(x_attr, y_attr)
        self.logreg_attr_trained = True

        if len(y_ex) > 0:
            if False not in y_ex:
                print("WARNING GETTING OPP WORDS")
                for x in self._helper_get_opposite_words(self.data_ex['word'].values):
                    self.add_data('ex', x, False)
                y_ex = np.array(self.data_ex['y'].values, dtype=bool)

            x_ex = np.concatenate(self.data_ex['x'].values)
            self.logreg_ex = self.logreg_ex.fit(x_ex, y_ex)
        self.logreg_ex_trained = True

    def predict(self, relation, word, is_word=True):
        if self.logreg_attr is None or self.logreg_ex is None or word is None:
            return 0.0

        if is_word:
            word = self.get_vector(word)

        if not self.logreg_attr_trained or not self.logreg_ex_trained:
            print("TRAINING")
            self.train()

        if relation == 'attr' and len(self.data_attr.index) > 0:
            return self.logreg_attr.predict_proba(word.reshape(1, -1))[0][1]
        elif relation == 'ex' and len(self.data_ex.index) > 0:
            return self.logreg_ex.predict_proba(word.reshape(1, -1))[0][1]
        else:
            return 0.0

    def predict_vectors(self, relation, vectors):
        return np.percentile([self.predict(relation, v, is_word=False) for v in vectors], 75)

    def predict_desc(self, relation, desc):
        predictions = []
        desc = self._clean_text(desc)
        for word in desc.split(' '):
            predictions.append(self.predict(relation, word))
            # print(w, self.predict_word(w))
        return np.percentile(np.array(predictions), 90)


if __name__ == "__main__":
    from gensim.models import KeyedVectors
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=200000)
    print("START")
    a = AbstractNode(model, 'happy')
    a.add_desc('attr', "delighted, pleased, or glad, as over a particular thing:", True)
    # a.add_desc('attr', "feeling or showing sorrow; unhappy.", False)
    print("ATTRIBUTES\n", a.data_attr)
    print("EXAMPLES\n", a.data_ex)
    print(a.predict_desc('attr', 'Life is like a roller coaster, live it, be happy, enjoy life. '))
    print("ATTRIBUTES\n", a.data_attr[['y', 'word']].head(10))
    print("EXAMPLES\n", a.data_ex[['y', 'word']].head(10))
    words = ['sad', 'happy', 'cat', 'feeling']
    for w in words:
        print(w, 'is attr of', a.name, ": ", a.predict('attr', w))
        print(w, 'is ex of', a.name, ": ", a.predict('ex', w))

