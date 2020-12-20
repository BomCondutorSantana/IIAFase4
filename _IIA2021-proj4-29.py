#!/usr/bin/env python
# coding: utf-8

# ## Projeto nº 4: Aprendizagem Automática
# 
# ### Introdução à Inteligência Artificial 2020/2021

# ### Etapa 1:
# Para carregar os dados é utilizada a função load_data fornecida no módulo utilsAA

# In[1]:


from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris # conjunto de dados
from sklearn.tree import DecisionTreeClassifier, plot_tree # árvore de decisão
from sklearn.neighbors import KNeighborsClassifier # k-NN
from sklearn.model_selection import train_test_split, cross_val_score # cross-validation
import numpy as np 
import matplotlib.pyplot as plt # gráficos
from utilsAA import * # módulo distribuido com o guião com funções auxiliares
from sklearn import preprocessing

airline_data, airline_target, airline_dclass, airline_tclass = load_data('airline.csv')


# ## Etapa 2:

# Começamos por fazer uma decision tree primeiro
# 
# Para passar de String a inteiros o que fazemos é o seguinte: vamos buscar a coluna i, que são todos elementos de uma classe de dados, vemos quantos valores diferentes existem (utilizando o unique) e guardamos os vários valores diferentes num array. De seguida vamos ver em que posição se encontra cada um desses elementos da nossa data no nosso array obtido pelo unique. A posição em que ele se encontra passa a ser o valor que vai ficar. Por exemplo: em relação a male e female, o array obtido pelo unique tem male na posição 1 e female na posição 0, portanto todos os males são substituídos 1 e female por 0.
# 
# Como a nosso ver o ID do cliente não influencia a sua classificação decidimos remover a coluna com os ID's dos clientes (np.delete(array, subarray, 0 ou 1 caso seja linha ou coluna)
# 

# In[378]:


for i in range(len(airline_data[0])):
    if type(airline_data[0][i]) == str:
        values = np.unique(airline_data[:,i])
        for j in range(len(airline_data)):
            airline_data[j][i] = np.where(values == airline_data[j][i])[0][0]   
airline_tdata = airline_data
airline_tdata = np.delete(airline_data, 0, 1)


# ## Etapa 3

# ### Melhor modelo
# Para descobrirmos o melhor modelo testámos os 

# In[379]:


scores = None
best = None
best_i = 0
best_j = 0
best_k = 0
for i in range(1,10):
    for j in range(2,10):
        for k in range(1,10):
            dtc = DecisionTreeClassifier(criterion='entropy', max_depth = i, min_samples_split = j, 
                                         min_samples_leaf = k)
            
            scores = cross_val_score(dtc, X=airline_tdata, y=airline_target, cv=10, n_jobs=-1)
            t = np.mean(scores)
            if best == None or t > best:
                best = t
                best_i = i
                best_j = j
                best_k = k

print(best_i)
print(best_j)
print(best_k)
dtc = DecisionTreeClassifier(criterion='entropy', max_depth = best_i, min_samples_split = best_j, 
                                         min_samples_leaf = best_k)

scores = cross_val_score(dtc, X=airline_tdata, y=airline_target, cv=10, n_jobs=-1)
print(np.std(scores))
print(np.mean(scores))


# In[4]:


9
3
1
0.039686445452274136
0.9010656010656011


# K-VIZINHOS

# In[370]:


scores = None
best = None
best_i = 0
best_erro = None
for i in range(1, 100): 
    clf = KNeighborsClassifier(n_neighbors=i)
    
    scores = cross_val_score(clf, X=airline_data, y=airline_target, cv=10, n_jobs=-1)
    t = np.mean(scores)
    if best == None or t > best:
        best = t
        best_i = i
        best_erro = np.std(scores)

clf = KNeighborsClassifier(n_neighbors=best_i)

scores = cross_val_score(clf, X=airline_data, y=airline_target, cv=10, n_jobs=-1)
print(best_i)
print(np.mean(scores))
print(np.std(scores))


# ### Previsões
# 

# In[14]:


test_data, test_dclass = load_data('test.csv', True)
#test_tdata = np.delete(test_data, 0, 1)
for i in range(len(test_tdata[0])):
    if type(test_tdata[0][i]) == str:
        values = np.unique(test_tdata[:,i])
        for j in range(len(test_tdata)):
            test_tdata[j][i] = np.where(values == test_tdata[j][i])[0][0]

predict = dtc.predict(test_tdata)
print(predict)

save_data("IIA2021-proj4-29.csv", predict)


# In[ ]:




