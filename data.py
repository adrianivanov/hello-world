import pandas as pd
dataset = pd.read_csv("tryharder1.2.csv")

#handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = "Comm Fail", strategy = 'constant', fill_value = 0)
imputer2 = SimpleImputer(missing_values = "Configure", strategy = 'constant', fill_value = 0)
imputer3 = SimpleImputer(missing_values = "I/O Timeout", strategy = 'constant', fill_value = 0)
imputer4 = SimpleImputer(missing_values = "ISU Saw No Data", strategy = 'constant', fill_value = 0)
imputer5 = SimpleImputer(missing_values = "Bad", strategy = 'constant', fill_value = 0)
imputer6 = SimpleImputer(missing_values = "Out of Serv", strategy = 'constant', fill_value = 0)
imputer7 = SimpleImputer(missing_values = "Shutdown", strategy = 'constant', fill_value = 0)
imputer8 = SimpleImputer(missing_values = "Bad Input", strategy = 'constant', fill_value = 0)
imputer9 = SimpleImputer(missing_values = "Equip Fail", strategy = 'constant', fill_value = 0)
imputer10 = SimpleImputer(missing_values = "Unit Down", strategy = 'constant', fill_value = 0)


imputer = imputer.fit(dataset)
imputer2 = imputer2.fit(dataset)
imputer3 = imputer3.fit(dataset)
imputer4 = imputer4.fit(dataset)
imputer5 = imputer5.fit(dataset)
imputer6 = imputer6.fit(dataset)
imputer7 = imputer7.fit(dataset)
imputer8 = imputer8.fit(dataset)
imputer9 = imputer9.fit(dataset)
imputer10 = imputer10.fit(dataset)


dataset = imputer.transform(dataset)
dataset = imputer2.transform(dataset)
dataset = imputer3.transform(dataset)
dataset = imputer4.transform(dataset)
dataset = imputer5.transform(dataset)
dataset = imputer6.transform(dataset)
dataset = imputer7.transform(dataset)
dataset = imputer8.transform(dataset)
dataset = imputer9.transform(dataset)
dataset = imputer10.transform(dataset)


dataset = pd.DataFrame({'amine':dataset[:,0], 'sws':dataset[:,1], 'gn':dataset[:,2], 'aer':dataset[:,3], 't soba':dataset[:,4], 't s1':dataset[:,5], 'so2':dataset[:,6], 'h2s':dataset[:,7], 'pres':dataset[:,8]})
x_aer_unsc = dataset.iloc[:, 0:3].to_numpy()

y_aer_unsc = dataset.iloc[:,3].to_numpy()
y_soba_unsc = dataset.iloc[:,4].to_numpy()
y_s1_unsc = dataset.iloc[:,5].to_numpy()
y_so2_unsc = dataset.iloc[:,6].to_numpy()
y_h2s_unsc = dataset.iloc[:,7].to_numpy()
y_pres_unsc = dataset.iloc[:,8].to_numpy()


from sklearn.preprocessing import StandardScaler
sc_x_aer = StandardScaler()
x_aer = sc_x_aer.fit_transform(x_aer_unsc.reshape(-1,3))
sc_y_aer = StandardScaler()
y_aer = sc_y_aer.fit_transform(y_aer_unsc.reshape(-1,1))
sc_soba = StandardScaler()
y_soba = sc_soba.fit_transform(y_soba_unsc.reshape(-1,1))
sc_s1 = StandardScaler()
y_s1 = sc_s1.fit_transform(y_s1_unsc.reshape(-1,1))
sc_so2 = StandardScaler()
y_so2 = sc_so2.fit_transform(y_so2_unsc.reshape(-1,1))
sc_h2s = StandardScaler()
y_h2s = sc_h2s.fit_transform(y_h2s_unsc.reshape(-1,1))
sc_pres = StandardScaler()
y_pres = sc_pres.fit_transform(y_pres_unsc.reshape(-1,1))

