from data import *
from sklearn.svm import SVR
import pickle

svr_aer = SVR(kernel = 'rbf')
svr_aer.fit(x_aer,y_aer.ravel())

svr_soba = SVR(kernel = 'rbf')
svr_soba.fit(x_aer,y_soba.ravel())

svr_s1 = SVR(kernel = 'rbf')
svr_s1.fit(x_aer,y_s1.ravel())

svr_so2 = SVR(kernel = 'rbf')
svr_so2.fit(x_aer,y_so2.ravel())

svr_h2s = SVR(kernel = 'rbf')
svr_h2s.fit(x_aer, y_h2s.ravel())

svr_pres = SVR(kernel = 'rbf')
svr_pres.fit(y_aer,y_pres.ravel())

def get_scores_svr():
    scor1 = svr_aer.score(x_aer, y_aer)
    scor2 = svr_aer.score(x_aer, y_soba)
    scor3 = svr_aer.score(x_aer, y_s1)
    scor4 = svr_aer.score(x_aer, y_so2)
    scor5 = svr_aer.score(x_aer, y_h2s)
    scor6 = svr_aer.score(x_aer, y_pres)
    return [scor1, scor2, scor3, scor4, scor5, scor6]
li_svr = get_scores_svr()

#making pickles
with open("svr_aer.pickle", "wb") as pickleaer:
            pickle.dump(svr_aer, pickleaer)
with open("svr_h2s.pickle", "wb") as pickleh2s:
            pickle.dump(svr_h2s, pickleh2s)
with open("svr_so2.pickle", "wb") as pickleso2:
            pickle.dump(svr_so2, pickleso2)
with open("svr_s1.pickle", "wb") as pickles1:
            pickle.dump(svr_s1, pickles1)
with open("svr_soba.pickle", "wb") as picklesoba:
            pickle.dump(svr_soba, picklesoba)
with open("svr_pres.pickle", "wb") as picklepres:
            pickle.dump(svr_pres, picklepres)
with open("li_svr.pickle", "wb") as picklescorsvr:
            pickle.dump(li_svr, picklescorsvr)