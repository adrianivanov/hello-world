import pickle
from data import *


#making and fitting the models
from sklearn.linear_model import LinearRegression

linear_aer = LinearRegression()
linear_aer.fit(x_aer, y_aer)

linear_soba = LinearRegression()
linear_soba.fit(x_aer, y_soba.ravel())

linear_s1 = LinearRegression(fit_intercept = True, normalize = True)
linear_s1.fit(x_aer, y_s1.ravel())

linear_so2 = LinearRegression()
linear_so2.fit(x_aer,y_so2.ravel())

linear_h2s = LinearRegression()
linear_h2s.fit(x_aer,y_h2s.ravel())

linear_pres = LinearRegression()
linear_pres.fit(y_aer, y_pres.ravel())

def get_scores_linear():
    scor1 = linear_aer.score(x_aer, y_aer)
    scor2 = linear_aer.score(x_aer, y_soba)
    scor3 = linear_aer.score(x_aer, y_s1)
    scor4 = linear_aer.score(x_aer, y_so2)
    scor5 = linear_aer.score(x_aer, y_h2s)
    scor6 = linear_aer.score(x_aer, y_pres)
    return [scor1, scor2, scor3, scor4, scor5, scor6]
li_linear = get_scores_linear()

#making pickles
with open("linear_aer.pickle", "wb") as pickleaer:
            pickle.dump(linear_aer, pickleaer)
with open("linear_h2s.pickle", "wb") as pickleh2s:
            pickle.dump(linear_h2s, pickleh2s)
with open("linear_so2.pickle", "wb") as pickleso2:
            pickle.dump(linear_so2, pickleso2)
with open("linear_s1.pickle", "wb") as pickles1:
            pickle.dump(linear_s1, pickles1)
with open("linear_soba.pickle", "wb") as picklesoba:
            pickle.dump(linear_soba, picklesoba)
with open("linear_pres.pickle", "wb") as picklepres:
            pickle.dump(linear_pres, picklepres)
with open("li_linear.pickle", "wb") as picklescorlinear:
            pickle.dump(li_linear, picklescorlinear)