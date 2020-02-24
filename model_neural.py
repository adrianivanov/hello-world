from data import *
import pickle
from sklearn.neural_network import MLPRegressor


neural_aer = MLPRegressor(hidden_layer_sizes = (10,5,2), activation = 'tanh')
neural_aer.fit(x_aer,y_aer.ravel())

neural_soba = MLPRegressor(hidden_layer_sizes = (10,5,2), activation = 'logistic', max_iter = 300)
neural_soba.fit(x_aer,y_soba.ravel())

neural_s1 = MLPRegressor(hidden_layer_sizes = (10,5,2), activation = 'tanh')
neural_s1.fit(x_aer,y_s1.ravel())

neural_so2 = MLPRegressor(hidden_layer_sizes = (10,5,2), activation = 'tanh')
neural_so2.fit(x_aer,y_so2.ravel())

neural_h2s = MLPRegressor(hidden_layer_sizes = (10,5,2), activation = 'tanh', max_iter = 300)
neural_h2s.fit(x_aer, y_h2s.ravel())

neural_pres = MLPRegressor(activation = 'tanh')
neural_pres.fit(y_aer,y_pres.ravel())

def get_scores_neural():
    scor1 = neural_aer.score(x_aer, y_aer)
    scor2 = neural_aer.score(x_aer, y_soba)
    scor3 = neural_aer.score(x_aer, y_s1)
    scor4 = neural_aer.score(x_aer, y_so2)
    scor5 = neural_aer.score(x_aer, y_h2s)
    scor6 = neural_aer.score(x_aer, y_pres)
    return [scor1, scor2, scor3, scor4, scor5, scor6]
li_neural = get_scores_neural()

#making pickles
with open("neural_aer.pickle", "wb") as pickleaer:
            pickle.dump(neural_aer, pickleaer)
with open("neural_h2s.pickle", "wb") as pickleh2s:
            pickle.dump(neural_h2s, pickleh2s)
with open("neural_so2.pickle", "wb") as pickleso2:
            pickle.dump(neural_so2, pickleso2)
with open("neural_s1.pickle", "wb") as pickles1:
            pickle.dump(neural_s1, pickles1)
with open("neural_soba.pickle", "wb") as picklesoba:
            pickle.dump(neural_soba, picklesoba)
with open("neural_pres.pickle", "wb") as picklepres:
            pickle.dump(neural_pres, picklepres)
with open("li_neural.pickle", "wb") as picklescorneural:
            pickle.dump(li_neural, picklescorneural)
            
            
            
            
            
        