#MÓDULO DE TREINAMENTO
#Clusterizador KMeans para a base Boston Housing

import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from sklearn.cluster import KMeans
from pickle import dump

#1 Abrir os dados
dados = pd.read_csv('HousingData.csv', sep=',')

#2 Tratar os valores ausentes (NA)
#A base possui NA nas colunas CRIM, ZN, INDUS, CHAS, AGE e LSTAT e o KMeans não aceita NA
#numéricos recebem a mediana da coluna; categóricos recebem a moda (valor mais frequente)

#3 NORMALIZAR OS DADOS
#Separar as colunas categóricas das numéricas
#CHAS é a única categórica (dummy: 1 = o setor margeia o rio Charles, 0 = não margeia)
dados_num = dados.drop(columns=['CHAS'])
dados_cat = dados['CHAS']
dados_cat_rotulos = 'CHAS' #se houver mais colunas categóricas, basta preencher a lista

dados_num = dados_num.fillna(dados_num.median())
dados_cat = dados_cat.fillna(dados_cat.mode()[0]).astype(int)

#3.1 Normalizar os dados numéricos
#Instanciar o normalizador
normalizador = preprocessing.MinMaxScaler()
#Treinar o modelo normalizador a partir dos dados numéricos
modelo_normalizador = normalizador.fit(dados_num)
#Salvar o modelo normalizador para uso posterior
dump(modelo_normalizador, open('normalizador_boston.model', 'wb'))
#Normalizar os dados numéricos
dados_num_normalizados = modelo_normalizador.transform(dados_num)
#recompor o dataframe com os dados numéricos normalizados, atribuindo os rótulos de colunas originais
dados_num_normalizados = pd.DataFrame(dados_num_normalizados, columns=dados_num.columns)

#3.2 Normalizar os dados categóricos
dados_cat_normalizados = pd.get_dummies(dados_cat, prefix=dados_cat_rotulos, dummy_na=False, dtype=int)
#Salvar as colunas geradas pelo get_dummies (CHAS_0, CHAS_1) para normalizar a nova residência na inferência
dump(list(dados_cat_normalizados.columns), open('colunas_categoricas_boston.model', 'wb'))

#3.3 Juntar os dados numéricos e categóricos normalizados
dados_normalizados = dados_num_normalizados.join(dados_cat_normalizados)

#4 BALANCEAR OS DADOS
#Não há necessidade de balancear a base (clusterização não tem classe)

#5 Descobrir o número ideal de clusters (de grupos)
#HIPERPARAMETRIZAÇÃO
#lembrando que estamos determinando, por enquanto, um hiperparâmetro:
#   número de clusters

distorcoes = [] #matriz para conter as distorções calculadas
K = range(1, 100) #K = range(1, dados.shape[0]) ==> dados.shape[0] retorna o número de registros em dados
for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42).fit(dados_normalizados)
    distorcoes.append(
        sum(
            np.min(
                cdist(dados_normalizados, cluster_model.cluster_centers_, 'euclidean'),
                axis=1)/dados_normalizados.shape[0]
            )
        )

#Plotar o gráfico das distorções
fig, ax = plt.subplots()
ax.plot(K, distorcoes)
ax.set(xlabel='n Clusters', ylabel='Distorcoes')
ax.grid()
fig.savefig('cotovelo.png')
# plt.show()

#Determinar o número ideal de grupos
#MÉTODO DO COTOVELO (ELBOW)
x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]
distancias = []
for i in range(len(distorcoes)):
    x = K[i]
    y = distorcoes[i]
    numerador = abs(
        (yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0
    )
    denominador = math.sqrt(
        (yn-y0)**2 + (xn-x0)**2
    )
    distancias.append(numerador/denominador)

n_clusters_otimo = K[distancias.index(max(distancias))]
print('Número ótimo de clusters:', n_clusters_otimo)

#6 Treinamento definitivo
boston_cluster = KMeans(n_clusters=n_clusters_otimo, random_state=42).fit(dados_normalizados)

#Salvar o modelo para uso posterior
dump(
    boston_cluster,
    open('boston_cluster.model', 'wb')
)
