from data import *
import pickle
from sklearn.ensemble import RandomForestRegressor


forest_aer = RandomForestRegressor(n_estimators = 10)
forest_aer.fit(x_aer, y_aer.ravel())

forest_soba = RandomForestRegressor(n_estimators = 10)
forest_soba.fit(x_aer, y_soba.ravel())

forest_s1 = RandomForestRegressor(n_estimators = 10)
forest_s1.fit(x_aer, y_s1.ravel())

forest_so2 = RandomForestRegressor(n_estimators = 10)
forest_so2.fit(x_aer,y_so2.ravel())

forest_h2s = RandomForestRegressor(n_estimators = 10)
forest_h2s.fit(x_aer,y_h2s.ravel())

forest_pres = RandomForestRegressor(n_estimators = 10)
forest_pres.fit(y_aer, y_pres.ravel())

def get_scores_forest():
    scor1 = forest_aer.score(x_aer, y_aer)
    scor2 = forest_aer.score(x_aer, y_soba)
    scor3 = forest_aer.score(x_aer, y_s1)
    scor4 = forest_aer.score(x_aer, y_so2)
    scor5 = forest_aer.score(x_aer, y_h2s)
    scor6 = forest_aer.score(x_aer, y_pres)
    return [scor1, scor2, scor3, scor4, scor5, scor6]
li_forest = get_scores_forest()

#making pickles
with open("forest_aer.pickle", "wb") as pickleaer:
            pickle.dump(forest_aer, pickleaer)
with open("forest_h2s.pickle", "wb") as pickleh2s:
            pickle.dump(forest_h2s, pickleh2s)
with open("forest_so2.pickle", "wb") as pickleso2:
            pickle.dump(forest_so2, pickleso2)
with open("forest_s1.pickle", "wb") as pickles1:
            pickle.dump(forest_s1, pickles1)
with open("forest_soba.pickle", "wb") as picklesoba:
            pickle.dump(forest_soba, picklesoba)
with open("forest_pres.pickle", "wb") as picklepres:
            pickle.dump(forest_pres, picklepres)
with open("li_forest.pickle", "wb") as picklescorforest:
            pickle.dump(li_forest, picklescorforest)