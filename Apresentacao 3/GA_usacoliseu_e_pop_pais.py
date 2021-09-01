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
objetivos_GA = {}
tempos = {}
# lista_hs = [0.8, 0.6, 0.4, 0.2]
lista_hs = [0.4]
# conjuntos = [10,20,50,100,200,500,1000]
conjuntos = [50]
lista_z = [0.25 , 0.5 , 0.6 , 0.75, 2]
lista_problemas = list(range(1,11))
lista_problemas = [9]

populacao = {}
qtd_pop_inicial = {}

for repeticoes in range(5):
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
                    #print('##1')
                    ini = time.time()

                    step1 = time.time()
                    



                    if usa_coliseu == 1:
                        populacao_filhos = []
                        # print('##2')
                        
                        # if iter <= n_iter_ga/2:
                        #     n_pop = int(n_pop_inicial + iter/2)
                        # else:
                        #     n_pop = int(n_pop_inicial + n_iter_ga/2 - iter/2)
                        
                        
                        #print("Step 1 - Coliseu: ", ini - step1)

                        populacao_filhos = [gerar_filho(populacao_pais, num_pais_duelo, ai, bi, pi, d) for _ in range(0, (int(n_pop*2) - len(populacao_total)))]
                    if usa_coliseu == 0:
                        
                        #print("Step 1 - Roleta: ", ini - step1, len(populacao_pais), len(populacao_fitness))
                        populacao_filhos = gerar_filho_roleta(populacao_total,populacao_fitness,ai,bi,pi,d)

                    step2 = time.time()
                    #print("Step 2: ", step2 - step1, len(populacao_filhos))

                    populacao_filhos_mutados = [mutacao(filho, taxa_mutacao, ai, bi, pi, d) for filho in populacao_filhos]
                        
                    step3 = time.time()
                    #print("Step 3: ", step3 - step2)

                    populacao_total = np.vstack((populacao_total,populacao_filhos_mutados))
                    populacao_fitness = np.hstack((populacao_fitness,[calcula_objetivo_GA(filho, ai, bi, pi, d)[1] for filho in populacao_filhos_mutados]))
                    
                    step4 = time.time()
                    #print("Step 4: ", step4 - step3)
                    # cromossomos_unicos = np.array([list(map(np.array, set(map(tuple, populacao_total))))])
                    # cromossomos_unicos = np.unique(populacao_total)
                    # cromossomos_unicos = populacao_total

                    step5 = time.time()


                    sobreviventes = oprime_fracos(populacao_fitness, n_pop, taxa_elitismo)
                    
                    step7 = time.time()
                    #print("Step 7: ", step7 - step5)

                    cromossomos_sobreviventes = populacao_total[sobreviventes==True]
                    fitness_sobreviventes = populacao_fitness[sobreviventes==True]

                    fitness_sobreviventes_sorted = np.flip(np.argsort(fitness_sobreviventes)[::-1])

                    cromossomos_sobreviventes = cromossomos_sobreviventes[fitness_sobreviventes_sorted]
                    fitness_sobreviventes = fitness_sobreviventes[fitness_sobreviventes_sorted]

                    step8 = time.time()
                    #print("Step 8: ", step8 - step7)
                    if fitness_sobreviventes[0] < melhor_obj:
                        melhor_obj, melhor_sol = fitness_sobreviventes[0], cromossomos_sobreviventes[0]
                        print(melhor_obj)

                    step9 = time.time()
                    #print("Step 9: ", step9 - step8)
                    populacao_total = cromossomos_sobreviventes
                    populacao_fitness = fitness_sobreviventes
                    populacao_pais = populacao_total[:max(2,int(perc_pais_pop*n_pop)+1)]
                    
                    step10 = time.time()
                    #print("Step 10: ", step10 - step9)
                    cromossomos_unicos = list(map(np.array, set(map(tuple, populacao_total))))
                    filhos_unicos = list(map(np.array, set(map(tuple, populacao_filhos))))
                    pais_unicos = list(map(np.array, set(map(tuple, populacao_pais))))
                    if iter%100 ==0: 
                        print(conjunto, h, problema, iter,len(populacao_total),len(filhos_unicos)/len(populacao_filhos),len(pais_unicos)/len(populacao_pais), len(cromossomos_unicos)/len(populacao_total), obj_original, populacao_fitness[0])
                    
                    #### Comentei a parte que reseta a aleatoriedade... se for reativar, precisa ajustar algum bug que ainda tá dando quando chama a função oprime_fracos
                    # if len(cromossomos_unicos)/len(populacao_total) < 0.2 and iter > 10:
                    #     print("Aleatoriedade abaixo de 20%")
                    #     # breakpoint()
                    #     populacao_total = np.vstack((populacao_total[:max(2,int(perc_pais_pop*n_pop/2)+1)],lista_populacao_inicial))#np.array([cromossomo for cromossomo in cromossomos_unicos])
                    #     populacao_fitness = np.array([calcula_objetivo_GA(cromossomo, ai, bi, pi, d)[1] for cromossomo in cromossomos_unicos])
                    #     populacao_pais = populacao_total[:max(2,int(perc_pais_pop*n_pop)+1)]


                fim = time.time()
                
                print("Fim: ", fim - step8)
                # print("Fim: ", fim-ini)
                # breakpoint()
                # print(len(populacao_total))
                solucoes_GA[(conjunto, h, problema,repeticoes)] = melhor_sol
                objetivos_GA[(conjunto, h, problema,repeticoes)] = melhor_obj
                tempos[(conjunto, h, problema,repeticoes)] = fim-ini
                    
                    




                #breakpoint()
                # for iteracoes in range(1,100):

                # for cromossomo_inicial in lista_populacao_inicial:
                    
                #     for i in range(1, 51):
                #         populacao_total.append(mutacao(cromossomo_inicial, 0.3))
objetivos_pandas = pd.Series(objetivos_GA)
tempos_pandas = pd.Series(tempos)

report = pd.ExcelWriter('resultados_GA.xlsx')

objetivos_unstack = objetivos_pandas.unstack(level=-3)
objetivos_unstack.to_excel(report, sheet_name=("Objetivos"))

t_unstack = tempos_pandas.unstack(level=-3)
t_unstack.to_excel(report,sheet_name="Tempos")

report.save()


                














