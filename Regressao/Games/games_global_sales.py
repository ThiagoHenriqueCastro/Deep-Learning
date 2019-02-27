from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
import pandas as pd
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor
from keras import optimizers

adam = optimizers.Adam(lr = 0.0001)

base = pd.read_csv('games.csv')
base = base.drop('Other_Sales', axis = 1)
base = base.drop('Developer', axis = 1)
base = base.drop('NA_Sales', axis = 1)
base = base.drop('EU_Sales', axis = 1)
base = base.drop('JP_Sales', axis = 1)
base = base.loc[base['Global_Sales'] > 1]
base = base.dropna(axis = 0)
nome_jogos = base.Name
base = base.drop('Name', axis = 1)

previsores = base.iloc[:, [0, 1, 2, 3, 5, 6, 7, 8, 9]].values
global_sales = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
previsores[:, 0] = labelencoder.fit_transform(previsores[:, 0])
previsores[:, 2] = labelencoder.fit_transform(previsores[:, 2])
previsores[:, 3] = labelencoder.fit_transform(previsores[:, 3])
previsores[:, 8] = labelencoder.fit_transform(previsores[:, 8])

onehotencoder = OneHotEncoder(categorical_features = [0, 2, 3, 8], n_values = 'auto')
previsores= onehotencoder.fit_transform(previsores).toarray()


camada_entrada = Input(shape=(99,))
camada_oculta1 = Dense(units = 75, activation = 'sigmoid')(camada_entrada)
camada_oculta2 = Dense(units = 75, activation = 'sigmoid')(camada_oculta1)
camada_saida = Dense(units = 1, activation = 'linear')(camada_oculta2)
    
regressor = Model(inputs = camada_entrada,
                      outputs = [camada_saida])
    
regressor.compile(optimizer = adam, 
                      loss = 'mean_squared_error')
    

regressor.fit(previsores, 
              global_sales,
              epochs = 10000,
              batch_size = 32)

previsao = regressor.predict(previsores)


regressor_json = regressor.to_json()
with open('regressor_games_global.json', 'w') as json_file:
    json_file.write(regressor_json)
regressor.save_weights('regressor_games_global.h5')

