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
solucoes_GA = {}

# lista_hs = [0.8, 0.6, 0.4, 0.2]
lista_hs = [0.4]
# conjuntos = [10,20,50,100,200,500,1000]
conjuntos = [1000]
lista_z = [0.25 , 0.5 , 0.6 , 0.75, 2]
lista_problemas = list(range(1,11))
lista_problemas = [9]

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
            obj_original = 0
            for individuo in lista_populacao_inicial:
                obj = calcula_objetivo(individuo, ai, bi, pi, d)
                
                if obj < melhor_obj:
                    melhor_obj, melhor_sol = obj, individuo
                    obj_original = melhor_obj



            populacao[(conjunto, h, problema)] = lista_populacao_inicial

            n_iter_ga = 2000
            taxa_mutacao_inicial = 1/conjunto
            n_pop_inicial = 500 #max(50,conjunto*0.1)
            perc_pais_pop = 0.5
            num_pais_duelo = 2
            taxa_elitismo = 0.75
            usa_coliseu = 0

            populacao_total = lista_populacao_inicial
            populacao_pais = lista_populacao_inicial

            
            fitness_populacao_inicial = np.array([calcula_objetivo(filho, ai, bi, pi, d) for filho in populacao_total])
            populacao_fitness = fitness_populacao_inicial
            n_pop = n_pop_inicial
            taxa_mutacao = taxa_mutacao_inicial
            for iter in range(n_iter_ga):
                # print('##1')
                ini = time.time()
                step1 = time.time()
                



                if usa_coliseu == 1:
                    populacao_filhos = []
                    # print('##2')
                    
                    # if iter <= n_iter_ga/2:
                    #     n_pop = int(n_pop_inicial + iter/2)
                    # else:
                    #     n_pop = int(n_pop_inicial + n_iter_ga/2 - iter/2)
                    
                    
                    # print("Step 1: ", ini - step1)

                    populacao_filhos = [gerar_filho(populacao_total, num_pais_duelo, ai, bi, pi, d) for _ in range(0, (int(n_pop*2) - len(populacao_total)))]
                if usa_coliseu == 0:
                    populacao_filhos = gerar_filho_roleta(populacao_total,populacao_fitness,ai,bi,pi,d)


                populacao_filhos_mutados = [mutacao(filho, taxa_mutacao, ai, bi, pi, d) for filho in populacao_filhos]
                    
                step2 = time.time()
                # print("Step 2: ", step2 - step1)

                populacao_total = np.vstack((populacao_total,populacao_filhos_mutados))
                populacao_fitness = np.hstack((populacao_fitness,[calcula_objetivo_GA(filho, ai, bi, pi, d)[1] for filho in populacao_filhos_mutados]))
                
                step3 = time.time()
                # print("Step 3: ", step3 - step2)
                # cromossomos_unicos = np.array([list(map(np.array, set(map(tuple, populacao_total))))])
                # cromossomos_unicos = np.unique(populacao_total)
                # cromossomos_unicos = populacao_total

                step5 = time.time()


                sobreviventes = oprime_fracos(populacao_fitness, n_pop, taxa_elitismo)
                
                step7 = time.time()

                cromossomos_sobreviventes = populacao_total[sobreviventes==True]
                fitness_sobreviventes = populacao_fitness[sobreviventes==True]





                step8 = time.time()
                # print("Step 8: ", step8 - step7)
                melhor_sobrevivente_obj = min(fitness_sobreviventes)
                if melhor_sobrevivente_obj < melhor_obj:
                    melhor_index = np.argmin(fitness_sobreviventes)
                    melhor_obj, melhor_sol = fitness_sobreviventes[melhor_index], cromossomos_sobreviventes[melhor_index]
                    print(melhor_obj)

                populacao_total = cromossomos_sobreviventes
                populacao_fitness = fitness_sobreviventes
                populacao_pais = populacao_total
                
                cromossomos_unicos = list(map(np.array, set(map(tuple, populacao_total))))
                filhos_unicos = list(map(np.array, set(map(tuple, populacao_filhos))))
                pais_unicos = list(map(np.array, set(map(tuple, populacao_pais))))
                print(conjunto, h, problema, iter,len(populacao_total),round(len(filhos_unicos)/len(populacao_filhos),3),round(len(pais_unicos)/len(populacao_pais),3), len(cromossomos_unicos)/len(populacao_total), obj_original, melhor_sobrevivente_obj)
                
                #### Comentei a parte que reseta a aleatoriedade... se for reativar, precisa ajustar algum bug que ainda t?? dando quando chama a fun????o oprime_fracos
                # if len(cromossomos_unicos)/len(populacao_total) < 0.2 and iter > 10:
                #     print("Aleatoriedade abaixo de 20%")
                #     # breakpoint()
                #     populacao_total = np.vstack((populacao_total[:max(2,int(perc_pais_pop*n_pop/2)+1)],lista_populacao_inicial))#np.array([cromossomo for cromossomo in cromossomos_unicos])
                #     populacao_fitness = np.array([calcula_objetivo_GA(cromossomo, ai, bi, pi, d)[1] for cromossomo in cromossomos_unicos])
                #     populacao_pais = populacao_total[:max(2,int(perc_pais_pop*n_pop)+1)]


                fim = time.time()
                # print("Fim: ", fim-ini)
                # breakpoint()
                # print(len(populacao_total))
                
                




            breakpoint()
            # for iteracoes in range(1,100):

            # for cromossomo_inicial in lista_populacao_inicial:
                
            #     for i in range(1, 51):
            #         populacao_total.append(mutacao(cromossomo_inicial, 0.3))


            














