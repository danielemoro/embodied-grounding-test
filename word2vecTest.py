from gensim.models import KeyedVectors
import numpy as np

# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=200000)

# # Deal with an out of dictionary word: Михаил (Michail)
# if 'Михаил' in model:
#     print(model['Михаил'].shape)
# else:
#     print('{0} is an out of dictionary word'.format('Михаил'))
#
# # Some predefined functions that show content related information for given words
# print(model.most_similar(positive=['woman', 'king']))
# print(model.most_similar(positive=['apple', 'type']))
# print(model.most_similar(positive=['apple'], negative=['fruit']))
#
# print(model.doesnt_match("breakfast cereal dinner lunch".split()))


def add_words(w1, w2):
    v1 = model.get_vector(w1)
    v2 = model.get_vector(w2)
    v3 = np.add(v1, v2)
    return model.similar_by_vector(v3)


def sub_words(w1, w2):
    v1 = model.get_vector(w1)
    v2 = model.get_vector(w2)
    v3 = np.subtract(v1, v2)
    return model.similar_by_vector(v3)


def sub_then_add_word(w1, w2, w3):
    v1 = model.get_vector(w1)
    v2 = model.get_vector(w2)
    v3 = model.get_vector(w3)
    diff = np.subtract(v1, v2)
    return model.similar_by_vector(np.add(v3, diff))


# print(sub_words("France", "Paris"))
# print(sub_then_add_word("France", "Paris", "Rome"))
# print(sub_then_add_word("Paris", "France", "Italy"))
print(sub_then_add_word("Sister", "Human", "Dog"))
print(sub_words("Paris", "City"))
print(add_words("Paris", "art"))
print(add_words("Paris", "Country"))

x = [i[0] for i in sub_words("Paris", "France")]
print(x)
print(model.most_similar(x))
# print(model.most_similar(["London, "]))

print(model.most_similar(positive=['Paris'], negative=['France']))
print(model.most_similar(positive=['France'], negative=['Paris']))

