import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle
from sklearn.naive_bayes import GaussianNB


#################### NAIVE BAYES ######################
base_risco_fraude = pd.read_csv(r'.\risco_fraude.csv')
#atributos previsores
x_risco_fraude = base_risco_fraude.iloc[:,0:4].values

#classe 
y_risco_fraude = base_risco_fraude.iloc[:,4].values

print(x_risco_fraude)
print(y_risco_fraude)

#transformando as strings em números
label_encoder_estabelecimento = LabelEncoder()
label_encoder_valor = LabelEncoder()
label_encoder_regiao = LabelEncoder()
label_encoder_limite = LabelEncoder()

x_risco_fraude[:,0] = label_encoder_estabelecimento.fit_transform(x_risco_fraude[:,0])
x_risco_fraude[:,1] = label_encoder_valor.fit_transform(x_risco_fraude[:,1])
x_risco_fraude[:,2] = label_encoder_regiao.fit_transform(x_risco_fraude[:,2])
x_risco_fraude[:,3] = label_encoder_limite.fit_transform(x_risco_fraude[:,3])

print(x_risco_fraude)

#salvando atributos transformados para utilizacoes futuras
with open('risco_credito.pkl', 'wb') as f:
    pickle.dump([x_risco_fraude, y_risco_fraude], f)

naive_risco_credito = GaussianNB()
naive_risco_credito.fit(x_risco_fraude, y_risco_fraude)

#submetendo novos dados
previsao = naive_risco_credito.predict([[2,0,2,2],[1,1,0,0],[0,0,1,1]])

#previsao
print("Previsão com base nos dados inputados: ") 
print(previsao)