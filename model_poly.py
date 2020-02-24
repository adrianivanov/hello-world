#this is not properly implemented yet as it has  issues with array dimensions and the predictions are way off

from data import *
import pickle
import numpy as np

from sklearn.linear_model import LinearRegression

linear_aer = LinearRegression()
linear_aer.fit(x_aer, y_aer)

linear_soba = LinearRegression()
linear_soba.fit(x_aer, y_soba.ravel())

linear_s1 = LinearRegression()
linear_s1.fit(x_aer, y_s1.ravel())

linear_so2 = LinearRegression()
linear_so2.fit(x_aer,y_so2.ravel())

linear_h2s = LinearRegression()
linear_h2s.fit(x_aer,y_h2s.ravel())

linear_pres = LinearRegression()
linear_pres.fit(y_aer, y_pres.ravel())

from sklearn.preprocessing import PolynomialFeatures

poly1_aer = PolynomialFeatures(degree = 5)
x_aer_poly = poly1_aer.fit_transform(x_aer)
poly1_aer.fit(x_aer_poly, y_aer)
poly2_aer = LinearRegression()
poly2_aer.fit(x_aer_poly, y_aer)

a = poly2_aer.predict(poly1_aer.fit_transform([[7500,1500,150]]))


poly1_soba = PolynomialFeatures(degree = 5)
poly2_soba = LinearRegression()
poly2_soba.fit(x_aer_poly, y_soba)

b = poly2_soba.predict(poly1_aer.fit_transform([[7500,1500,150]]))

poly1_s1 = PolynomialFeatures(degree = 5)
poly2_s1 = LinearRegression()
poly2_s1.fit(x_aer_poly, y_s1)

c = poly2_s1.predict(poly1_aer.fit_transform([[7500,1500,150]]))


poly1_so2 = PolynomialFeatures(degree = 5)
poly2_so2 = LinearRegression()
poly2_so2.fit(x_aer_poly, y_so2)

d = poly2_so2.predict(poly1_aer.fit_transform([[7500,1500,150]]))

poly1_h2s = PolynomialFeatures(degree = 5)
poly2_h2s = LinearRegression()
poly2_h2s.fit(x_aer_poly, y_h2s)

e = poly2_h2s.predict(poly1_aer.fit_transform([[7500,1500,150]]))


poly1_pres = PolynomialFeatures(degree = 5)
y_aer_poly = poly1_pres.fit_transform(y_aer)
poly1_pres.fit(y_aer_poly, y_pres.ravel())
poly2_pres = LinearRegression()
poly2_pres.fit(y_aer_poly, y_pres)


f = poly2_pres.predict(poly1_pres.fit_transform(a))
