from gensim.models import KeyedVectors

from abstract_node import AbstractNode


class AbstractBrain:

    def __init__(self):
        self.word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=200000)
        self.nodes = {}

    def add_node(self, name):
        new_node = AbstractNode(self.word2vec, name)
        self.nodes[new_node.name] = new_node

    def add_data_node(self, name, relation, pos_desc, neg_desc=None):
        node = self.nodes[name]
        if pos_desc: node.add_desc(relation, pos_desc, True)
        if neg_desc: node.add_desc(relation, neg_desc, False)

    def classify(self, names, vectors, abstraction=0, max_abstractions=5, relation='attr'):
        if abstraction > max_abstractions or len(vectors) < 1: return []
        results = []
        for name in self.nodes:
            if name not in names:
                results.append((abstraction, name, self.nodes[name].predict_vectors(relation, vectors),
                                self.nodes[name].name_vector))

        results = sorted(results, key=lambda x: x[2], reverse=True)
        # print([i[:3] for i in results])
        best_results = [i for i in results if 0.7 < i[2]][:1]
        best_results += self.classify([i[1] for i in best_results], [i[3] for i in best_results],
                                      abstraction=abstraction+1, relation=relation)
        return [i[:3] for i in best_results]

    def classify_from_desc(self, desc, relation='attr'):
        names = AbstractNode._clean_text(desc).split(' ')
        vectors = [self.get_vector(word) for word in names]
        return self.classify(names, vectors, relation=relation)

    def classify_pretty(self, desc):
        query = "[" +str(desc) + "] is an attribute of "
        mylist = self.classify_from_desc(desc, relation='attr')
        for i, a in enumerate(mylist):
            query += "[" + str(a[1]) + "]"
            if i + 1 < len(mylist):
                query += ", which is an attribute of "

        query = "\n[" + str(desc) + "] is an example of "
        mylist = self.classify_from_desc(desc, relation='ex')
        for i, a in enumerate(mylist):
            query += "[" + str(a[1]) + "]"
            if i + 1 < len(mylist):
                query += ", which is an example of "

        return query


    def get_vector(self, word):
        if word in self.word2vec.vocab:
            return self.word2vec[word].reshape(1, -1)
        else:
            print("WARNING", word, "not in w2v vocab")
            return None

if __name__ == "__main__":
    b = AbstractBrain()
    b.add_node("red")
    b.add_data_node("red", 'ex', 'rose fire apple', 'color sky water blueberry')
    b.add_node("rose")
    b.add_data_node("rose", 'attr', 'red', 'blue fire apple')
    b.add_node("fire")
    b.add_data_node("fire", 'attr', 'red', 'blue rose apple')
    b.add_node("apple")
    b.add_data_node("apple", 'attr', 'red', 'blue fire rose')
    b.add_node("blue")
    b.add_data_node("blue", 'ex', 'sky water blueberry', 'color fire apple rose')
    b.add_node("sky")
    b.add_data_node("sky", 'attr', 'blue', 'red water blueberry')
    b.add_node("water")
    b.add_data_node("water", 'attr', 'blue', 'red sky blueberry')
    b.add_node("blueberry")
    b.add_data_node("blueberry", 'attr', 'blue', 'red water sky')
    b.add_node("color")
    b.add_data_node("color", 'ex', 'red blue', 'sky water blueberry rose fire apple')

    from pprint import pprint
    print("red is an attribute of:")
    pprint(b.classify_from_desc('red', 'attr'))
    print("red is an example of:")
    pprint(b.classify_from_desc('red', 'ex'))
    print("rose is an example of:")
    pprint(b.classify_from_desc('rose', 'ex'))

    print(b.classify_pretty("rose"))
    print(b.classify_pretty("red"))
    print(b.classify_pretty("blue"))
    print(b.classify_pretty("sky"))

