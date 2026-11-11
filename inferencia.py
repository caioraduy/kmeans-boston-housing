#MÓDULO DE INFERÊNCIA
#Receber uma nova residência e determinar a qual cluster ela pertence

import pandas as pd
from pickle import load

#1 Receber os dados da nova residência
nova_residencia = {
    'CRIM': 0.05, 'ZN': 20, 'INDUS': 4.0, 'CHAS': 0, 'NOX': 0.45, 'RM': 6.8, 'AGE': 40,
    'DIS': 5.5, 'RAD': 4, 'TAX': 300, 'PTRATIO': 17, 'B': 390, 'LSTAT': 6.5, 'MEDV': 28
}
nova_residencia = pd.DataFrame([nova_residencia])

#2 Normalizar a nova residência
#Separar as colunas categóricas das numéricas (igual ao treinamento)
nova_residencia_num = nova_residencia.drop(columns=['CHAS'])
nova_residencia_cat = nova_residencia['CHAS']

#2.1 Normalizar os dados numéricos
#carregar o normalizador salvo durante o treinamento
normalizador = load(open('normalizador_boston.model', 'rb'))
nova_residencia_num_normalizada = normalizador.transform(nova_residencia_num)
nova_residencia_num_normalizada = pd.DataFrame(nova_residencia_num_normalizada, columns=nova_residencia_num.columns)

#2.2 Normalizar os dados categóricos
#carregar as colunas geradas pelo get_dummies no treinamento (CHAS_0, CHAS_1)
colunas_categoricas = load(open('colunas_categoricas_boston.model', 'rb'))
nova_residencia_cat_normalizada = pd.get_dummies(nova_residencia_cat, prefix='CHAS', dummy_na=False, dtype=int)
#uma residência só gera uma coluna (ex.: só CHAS_0); completar com as colunas do treinamento
nova_residencia_cat_normalizada = nova_residencia_cat_normalizada.reindex(columns=colunas_categoricas, fill_value=0)

#2.3 Juntar os dados normalizados
nova_residencia_normalizada = nova_residencia_num_normalizada.join(nova_residencia_cat_normalizada)

#3 Inferir o cluster da nova residência
#carregar o modelo
boston_cluster = load(open('boston_cluster.model', 'rb'))
cluster_da_residencia = boston_cluster.predict(nova_residencia_normalizada)
print('A nova residência pertence ao cluster:', cluster_da_residencia[0])

#4 Desnormalizar o centroide do cluster inferido
centroide = boston_cluster.cluster_centers_[cluster_da_residencia]
centroide = pd.DataFrame(centroide, columns=nova_residencia_normalizada.columns)

#4.1 Desnormalizar os dados numéricos usando o método inverse_transform do modelo normalizador
centroide_num = centroide[nova_residencia_num.columns]
centroide_num_natural = normalizador.inverse_transform(centroide_num)
centroide_num_natural = pd.DataFrame(centroide_num_natural, columns=nova_residencia_num.columns)

#4.2 Desnormalizar os dados categóricos (criados com o get_dummies)
#no centroide as dummies guardam a proporção de cada categoria no cluster (ex.: CHAS_0 = 0.97)
#a categoria de maior proporção recebe 1 e as demais 0, para poder usar o from_dummies
centroide_cat = centroide[colunas_categoricas]
centroide_cat = centroide_cat.eq(centroide_cat.max(axis=1), axis=0).astype(int)
centroide_cat_natural = pd.from_dummies(centroide_cat, sep='_')

#4.3 Juntar e exibir o centroide desnormalizado, na ordem original das colunas
centroide_natural = centroide_num_natural.join(centroide_cat_natural)
centroide_natural = centroide_natural[nova_residencia.columns]
print('Centroide do cluster', cluster_da_residencia[0], '(dados desnormalizados):')
print(centroide_natural.T)
