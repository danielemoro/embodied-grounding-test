import random
from abstract_brain import AbstractBrain
from abstract_node import AbstractNode
from pprint import pprint
import pickle
import os.path


to_learn = []
for w in reversed(['viola', "piano", 'horse', 'sheep', 'earth', 'person', 'car', 'plane']):
    to_learn.append(w)

def search_new_words(input_desc):
    for w in AbstractNode._clean_text(input_desc).split(' '):
        w = w.lower().strip()
        if w not in b.nodes and w not in to_learn:
            to_learn.append(w)
            random.shuffle(to_learn)

def get_input():
    x = input()
    if x == '-tl':
        pprint(to_learn)
        x = input()
    if x == '-al':
        pprint(b.nodes)
        x = input()
    return x

def ask_attributes(curr):
    print('What are the attributes or examples of ' + curr + '?')
    desc = get_input()
    desc_not = ""

    if len(b.nodes) > 2:
        rand_words = random.sample([*b.nodes], 2)
        for rw in rand_words:
            print('Is ' + rw + " an attribute or example of", curr)
            answer = any(x.lower() in ["yes", 'y', 'sure', 'true', 'yep', 'mostly', 'usually', 'kind'] for x in get_input())
            if answer:
                desc += " " + rw
            else:
                desc_not += " " + rw

    search_new_words(desc_not)
    search_new_words(desc)
    desc_not = None if desc_not == "" else desc_not
    return desc, desc_not

b = AbstractBrain()
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
    desc, desc_not = ask_attributes(curr)

    b.add_node(curr, desc, desc_not)

    discovered_attributes = b.classify_from_desc(curr)
    if len(discovered_attributes) > 0:
        try_again = True
        tries = 0
        while try_again and tries <= 2:
            tries += 1
            query = "So does that mean that " + curr + " is "
            mylist = b.classify_from_desc(curr)
            for i, a in enumerate(mylist):
                query += a[1]
                if i+1 < len(mylist):
                    query += " and " + a[1] + " is "
            print(query + "?")
            try_again = not any(x.lower() in ["yes", 'y', 'sure', 'true', 'yep'] for x in get_input())

            if try_again:
                print("\nOops ok. So " + curr + " is not " + mylist[0][1] + " and " + mylist[0][1] + " is not " + curr)
                b.modify_node(mylist[0][1], None, (curr+" ")*10)
                b.modify_node(curr, None, (mylist[0][1]+" ")*10)
    print(" ...fascinating")
