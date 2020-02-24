from data import *
import pickle
from sklearn.tree import DecisionTreeRegressor


tree_aer = DecisionTreeRegressor(random_state = 42)
tree_aer.fit(x_aer, y_aer)

tree_soba = DecisionTreeRegressor(random_state = 42)
tree_soba.fit(x_aer, y_soba.ravel())

tree_s1 = DecisionTreeRegressor(random_state = 42)
tree_s1.fit(x_aer, y_s1.ravel())

tree_so2 = DecisionTreeRegressor(random_state = 42)
tree_so2.fit(x_aer,y_so2.ravel())

tree_h2s = DecisionTreeRegressor(random_state = 42)
tree_h2s.fit(x_aer,y_h2s.ravel())

tree_pres = DecisionTreeRegressor(random_state = 42)
tree_pres.fit(y_aer, y_pres.ravel())

def get_scores_tree():
    scor1 = tree_aer.score(x_aer, y_aer)
    scor2 = tree_aer.score(x_aer, y_soba)
    scor3 = tree_aer.score(x_aer, y_s1)
    scor4 = tree_aer.score(x_aer, y_so2)
    scor5 = tree_aer.score(x_aer, y_h2s)
    scor6 = tree_aer.score(x_aer, y_pres)
    return [scor1, scor2, scor3, scor4, scor5, scor6]
li_tree = get_scores_tree()

#making pickles
with open("tree_aer.pickle", "wb") as pickleaer:
            pickle.dump(tree_aer, pickleaer)
with open("tree_h2s.pickle", "wb") as pickleh2s:
            pickle.dump(tree_h2s, pickleh2s)
with open("tree_so2.pickle", "wb") as pickleso2:
            pickle.dump(tree_so2, pickleso2)
with open("tree_s1.pickle", "wb") as pickles1:
            pickle.dump(tree_s1, pickles1)
with open("tree_soba.pickle", "wb") as picklesoba:
            pickle.dump(tree_soba, picklesoba)
with open("tree_pres.pickle", "wb") as picklepres:
            pickle.dump(tree_pres, picklepres)
with open("li_tree.pickle", "wb") as picklescortree:
            pickle.dump(li_tree, picklescortree)