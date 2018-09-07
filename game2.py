import random
from abstract_brain import AbstractBrain
from abstract_node import AbstractNode
from pprint import pprint
import pickle
import os.path

b = AbstractBrain()
to_learn = []
for w in reversed(['viola', "piano", 'horse', 'sheep', 'earth', 'person', 'car', 'plane']):
    to_learn.append(w)

def get_input():
    x = input()
    if x == '-tl':
        pprint(to_learn)
        x = input()
    if x == '-al':
        pprint(b.nodes)
        x = input()
    return x

def search_new_words(input_desc):
    for w in AbstractNode._clean_text(input_desc).split(' '):
        w = w.lower().strip()
        if w not in b.nodes and w not in to_learn:
            to_learn.append(w)
            random.shuffle(to_learn)

def ask_attributes(curr):
    print('What are the attributes of ' + curr + '? Separate with commas')
    attributes = [i.strip() for i in get_input().split(",")]

    # FIND NON ATTRIBUTES
    # print('What are NOT the attributes of ' + curr + '? Separate with commas')
    # attributes_not = [i.strip() for i in get_input().split(",")]
    attributes_not = []

    # DEVELOP EXISTING GRAPH
    if len(b.nodes) > 2:
        rand_words = random.sample([*b.nodes], 2)
        for rw in rand_words:
            print('Is ' + rw + " an attribute of", curr)
            answer = any(x.lower() in ["yes", 'y', 'sure', 'true', 'yep', 'mostly', 'usually', 'kind'] for x in get_input())
            if answer:
                attributes.append(rw)
            else:
                attributes_not.append(rw)

    search_new_words(" ".join(attributes))
    search_new_words(" ".join(attributes_not))

    # ADD NODES
    if curr not in b.nodes:
        b.add_node(curr, None, None)

    for a in attributes:
        if a not in b.nodes:
            b.add_node(a, curr, None)
        else:
            b.modify_node(a, curr, None)
    for a in attributes_not:
        if a not in b.nodes:
            b.add_node(a, None, curr)
        else:
            b.modify_node(a, None, curr)

print("Your baby is learning their first words and trying to make sense of the world. "
      "Aid them in understanding what certain objects mean")
print("Just so I remember you later, what is your name?")
name = input()+".p"
if os.path.isfile(name):
    print("Welcome back " + name)
    b, to_learn = pickle.load(open(name, "rb"))

while len(to_learn) > 0:
    pickle.dump((b, to_learn), open(name, "wb"))
    curr = to_learn.pop()
    print("%%%%%%%%%")
    ask_attributes(curr)

    discovered_attributes = b.classify_from_desc(curr)
    if len(discovered_attributes) > 0:
        try_again = True
        tries = 0
        while try_again and tries <= 2:
            tries += 1
            query = "So does that mean that " + curr + " has "
            mylist = b.classify_from_desc(curr)
            for i, a in enumerate(mylist):
                query += a[1]
                if i+1 < len(mylist):
                    query += " and " + a[1] + " has "
            print(query + "?")
            try_again = not any(x.lower() in ["yes", 'y', 'sure', 'true', 'yep'] for x in get_input())

            if try_again:
                print("\nOops ok. So " + curr + " doesn't have " + mylist[0][1])
                b.modify_node(mylist[0][1], None, curr)

    print(" ...fascinating")
