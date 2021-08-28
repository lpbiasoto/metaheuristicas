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
# lista_problemas = list(range(1,11))
lista_problemas = [3]

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

            n_iter_ga = 1000
            taxa_mutacao_inicial = 0.30
            n_pop_inicial = max(20,conjunto*0.1)

            populacao_total = lista_populacao_inicial
            populacao_pais = lista_populacao_inicial

            for iter in range(n_iter_ga):
                print('##1')
                ini = time.time()
                populacao_filhos = []

                taxa_mutacao = (n_iter_ga - iter)/n_iter_ga*(taxa_mutacao_inicial-0.01) + 0.01
                print('##2')

                if iter <= n_iter_ga/2:
                    n_pop = int(n_pop_inicial + iter/2)
                else:
                    n_pop = int(n_pop_inicial + n_iter_ga/2 - iter/2)
                print('##3')

                for i in range(0, (n_pop*2 - len(populacao_total))):
                    pai1 = coliseu(populacao_pais,2, ai, bi, pi, d)[0]
                    pai2 = coliseu(populacao_pais,2, ai, bi, pi, d)[0]
                    filho = crossover_r(pai1, pai2)
                    filho_mutacao = mutacao(filho, taxa_mutacao)
                    populacao_filhos.append(filho_mutacao)
                print('##4')
                
                populacao_total = np.vstack((populacao_total,populacao_filhos))
                
                print('##5')
                cromossomos_unicos = list(map(np.array, set(map(tuple, populacao_total))))
                
                print('##6')
                fitness_cromossomos = []
                cromossomos_reparados = []

                for cromossomo in cromossomos_unicos:
                    breakpoint()
                    print('##6.1', cromossomo)
                    solucao, obj = calcula_objetivo_GA(cromossomo, ai, bi, pi, d)
                    
                    print('##6.2', cromossomo)
                    fitness_cromossomos.append(obj)
                    cromossomos_reparados.append(solucao)
                
                print('##7')
                fitness_cromossomos = np.array(fitness_cromossomos)
                cromossomos_reparados = np.array(cromossomos_reparados)
                
                sobreviventes = oprime_fracos(fitness_cromossomos, n_pop)
                print('##8')

                cromossomos_unicos = cromossomos_reparados[sobreviventes==True]

                fitness_cromossomos = fitness_cromossomos[sobreviventes==True]
                fitness_cromossomos_sorted = np.flip(np.argsort(fitness_cromossomos)[::-1])
                cromossomos_unicos = cromossomos_unicos[fitness_cromossomos_sorted]
                fitness_cromossomos = fitness_cromossomos[fitness_cromossomos_sorted]

                if fitness_cromossomos[0] < melhor_obj:
                    melhor_obj, melhor_sol = fitness_cromossomos[0], cromossomos_unicos[0]
                    print(melhor_obj)
                print('##9')

                populacao_total = cromossomos_unicos
                
                print(iter,len(populacao_total), round(calcula_diversidade(cromossomos_unicos,fitness_cromossomos),2), melhor_obj, fitness_cromossomos[0])
                # print(len(populacao_total))
                populacao_pais = cromossomos_unicos[:max(2,int(0.1*n_pop))]

                fim = time.time()




            breakpoint()
            # for iteracoes in range(1,100):

            # for cromossomo_inicial in lista_populacao_inicial:
                
            #     for i in range(1, 51):
            #         populacao_total.append(mutacao(cromossomo_inicial, 0.3))


            














