from gensim.models import KeyedVectors

from abstract_node import AbstractNode


class AbstractBrain:

    def __init__(self):
        self.word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=200000)
        self.nodes = {}

    def add_node(self, name, pos_desc, neg_desc):
        new_node = AbstractNode(self.word2vec, name)
        if pos_desc: new_node.give_desc(pos_desc, True)
        if neg_desc: new_node.give_desc(neg_desc, False)

        self.nodes[new_node.name] = new_node

    def modify_node(self, name, pos_desc, neg_desc):
        node = self.nodes[name]
        if pos_desc: node.give_desc(pos_desc, True)
        if neg_desc: node.give_desc(neg_desc, False)

    def classify(self, names, vectors, abstraction=0, max_abstractions=5):
        if abstraction > max_abstractions or len(vectors) < 1: return []

        results = []
        for name in self.nodes:
            if name not in names:
                results.append((abstraction, name, self.nodes[name].predict_vectors(vectors), self.nodes[name].name_vector))

        results = sorted(results, key=lambda x: x[2], reverse=True)
        # print([i[:3] for i in results])
        best_results = [i for i in results if 0.7 < i[2]][:1]
        best_results += self.classify([i[1] for i in best_results], [i[3] for i in best_results], abstraction=abstraction+1)
        return [i[:3] for i in best_results]

    def classify_from_desc(self, desc):
        names = AbstractNode._clean_text(desc).split(' ')
        vectors = [self.get_vector(word) for word in names]
        return self.classify(names, vectors)

    def get_vector(self, word):
        if word in self.word2vec.vocab:
            return self.word2vec[word].reshape(1, -1)
        else:
            print("WARNING", word, "not in w2v vocab")
            return None

if __name__ == "__main__":
    b = AbstractBrain()
    b.add_node('happy', "delighted, pleased, or glad, as over a particular thing", "feeling or showing sorrow; unhappy")
    b.add_node('smile', "to assume a facial expression indicating pleasure, favor, or amusement", "to contract the brow, as in displeasure or deep thought; scowl.")
    b.add_node('person', 'a human being, whether an adult or child:', 'a small domesticated carnivore')
    b.add_node('cow', 'female herbivore that provides food', 'a small domesticated carnivore')
    b.add_node('group', 'many cows', 'single one few')
    b.add_node('livestock', 'a group of many animals', None)
    b.add_node('value', 'the importance of something. livestock', 'dirt nothing')
    b.add_node('wealth', 'aggregate value. money', 'lack of things')

    from pprint import pprint
    pprint(b.classify_from_desc('herbivore'))