import random
from abstract_brain import AbstractBrain
from abstract_node import AbstractNode
from pprint import pprint
import pickle
import os.path

b = AbstractBrain()
b, to_learn = pickle.load(open('bob3.p', "rb"))

print(b.classify_from_desc("rain"))
print(b.classify_from_desc("person"))
print(b.classify_from_desc("music"))
print(b.classify_from_desc("rain"))