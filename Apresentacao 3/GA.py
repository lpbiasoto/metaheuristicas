from pathlib import Path
import math
import os
import numpy as np
import pandas as pd
import time
from numba import jit
import pickle
from funcoes_BuscaLocal import *
from funcoes_GA import *
from funcoes_gerais import *

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

            lista_populacao_inicial = np.array([solucoes[(c,p,hi,z_corte)] for (c,p,hi,z_corte) in solucoes if conjunto == c and problema == p and h == hi])

            melhor_obj, melhor_sol = 99999999999, np.array([])

            for individuo in lista_populacao_inicial:
                obj = calcula_objetivo(individuo, ai, bi, pi, d)
                if obj < melhor_obj:
                    melhor_obj, melhor_sol = obj, individuo

            populacao[(conjunto, h, problema)] = lista_populacao_inicial
            # list_arrays_unicos = list(map(np.array, set(map(tuple, lista_populacao_inicial))))
            # qtd_pop_inicial[(conjunto, h, problema)] = len(list_arrays_unicos)

            n_iter_ga = 3000
            taxa_mutacao_inicial = 0.30
            n_pop_inicial = max(2,conjunto*0.1)

            populacao_total = lista_populacao_inicial
            # cromossomos_unicos = len(list(map(np.array, set(map(tuple, populacao_total)))))
            populacao_pais = []
            populacao_filhos = []
            populacao_pais = lista_populacao_inicial
            for iter in range(n_iter_ga):
                ini = time.time()
                populacao_filhos = []

                taxa_mutacao = (n_iter_ga - iter)/n_iter_ga*(taxa_mutacao_inicial-0.01) + 0.01

                if iter <= n_iter_ga/2:
                    n_pop = int(n_pop_inicial + iter/2)
                else:
                    n_pop = int(n_pop_inicial + n_iter_ga/2 - iter/2)

                pais = np.array([coliseu(populacao_pais,2, ai, bi, pi, d)[0] for _ in range(n_pop*2)])
                
                for i in range(0, n_pop, 2):
                    pai1, pai2 = pais[i], pais[i+1]
                    filho = crossover_r(pai1, pai2)
                    filho_mutacao = mutacao(filho, taxa_mutacao)
                    populacao_filhos.append(filho_mutacao)

                populacao_total = np.vstack((populacao_pais,populacao_filhos))

                cromossomos_unicos_pop = list(map(np.array, set(map(tuple, populacao_total))))

                # objs_pop, cromossomos_pop = avaliar_pop(cromossomos_unicos_pop, ai, bi, pi, d)

                fitness_cromossomos = []
                for cromossomo in cromossomos_unicos_pop:
                    fitness_cromossomos.append(calcula_objetivo_GA(cromossomo, ai, bi, pi, d))
                
                fitness_cromossomos = np.array(fitness_cromossomos)
                fim = time.time()
                print(iter, fim-ini)




            breakpoint()
            # for iteracoes in range(1,100):

            # for cromossomo_inicial in lista_populacao_inicial:
                
            #     for i in range(1, 51):
            #         populacao_total.append(mutacao(cromossomo_inicial, 0.3))


            














