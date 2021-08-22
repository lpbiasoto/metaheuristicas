from pathlib import Path
import math
import os
import numpy as np
import pandas as pd
import time
from numba import jit
import pickle
from funcoes_GA import *

dados = {}

with open("dados.pkl", "rb") as infile:
    dados = pickle.load(infile)

solucoes_pandas = pd.read_pickle('solucoes.pkl')
solucoes = solucoes_pandas.to_dict()

lista_hs = [0.8, 0.6, 0.4, 0.2]
conjuntos = [10,20,50,100,200,500,1000]
lista_z = [0.25 , 0.5 , 0.6 , 0.75, 2]
lista_problemas = list(range(1,11))




populacao = {}
qtd_pop_inicial = {}
for conjunto in conjuntos:
    for h in lista_hs:
        for problema in lista_problemas:
            inicio=time.time()
            pi = np.array(dados[conjunto][problema]['pi'])
            ai = np.array(dados[conjunto][problema]['ai'])
            bi = np.array(dados[conjunto][problema]['bi'])

            d=int(sum(pi)*h)

            lista_populacao_inicial = np.array([solucoes[(conjunto,problema,h,z_corte)] for z_corte in lista_z])
            populacao[(conjunto, h, problema)] = lista_populacao_inicial
            list_arrays_unicos = list(map(np.array, set(map(tuple, lista_populacao_inicial))))
            qtd_pop_inicial[(conjunto, h, problema)] = len(list_arrays_unicos)

            populacao_total = []
            
            cromossomos_unicos = len(list(map(np.array, set(map(tuple, populacao_total)))))
            
            # for iteracoes in range(1,100):

            # for cromossomo_inicial in lista_populacao_inicial:
                
            #     for i in range(1, 51):
            #         populacao_total.append(mutacao(cromossomo_inicial, 0.3))


            














