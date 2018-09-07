import random
from abstract_brain import AbstractBrain
from abstract_node import AbstractNode
from pprint import pprint
import pickle
import os.path

b = AbstractBrain()
b, to_learn = pickle.load(open('cow.p', "rb"))

print(b.classify_from_desc("cow"))
print(b.classify_from_desc("sheep"))
print(b.classify_from_desc("grass"))
print(b.classify_from_desc("ground"))