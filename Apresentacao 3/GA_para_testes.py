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
inicio_de_verdade = time.time()
dados = {}

with open("dados.pkl", "rb") as infile:
    dados = pickle.load(infile)

solucoes_pandas = pd.read_pickle('solucoes.pkl')
solucoes = solucoes_pandas.to_dict()
solucoes_GA = {}
objetivos_GA = {}
tempos = {}
lista_hs = [0.8, 0.6, 0.4, 0.2]
#lista_hs = [0.2]
#lista_hs = [0.4]
#conjuntos = [10,20,50,100,200,500,1000]
conjuntos = [200]
lista_z = [0.25 , 0.5 , 0.6 , 0.75, 2]
#lista_problemas = list(range(1,11))
lista_problemas = [3]

populacao = {}
qtd_pop_inicial = {}
n_pop_inicial = 500 #max(50,conjunto*0.1)
n_iter_ga = 1000
n_repeticoes = 1 
num_pais_duelo = 2

taxa_mutacao_inicial = 1
taxa_elitismo = 0.75
usa_coliseu = 1
perc_pais_pop = taxa_elitismo


@jit(nopython=True)
def algortimo_genetico(n_iter_ga,taxa_mutacao_inicial,n_pop_inicial,perc_pais_pop,num_pais_duelo,taxa_elitismo,usa_coliseu,ai,bi,pi,d,repeticoes,lista_populacao_inicial,conjunto,h,problema):  

    guarda_objetivos = np.array([1e10,1e10])
    
    objs = np.array([calcula_objetivo_sr(individuo, ai, bi, pi, d) for individuo in lista_populacao_inicial])
        
    melhor_obj_index = np.argmin(objs)
    melhor_sol = lista_populacao_inicial[melhor_obj_index]
    melhor_obj = objs[melhor_obj_index]

    populacao_total = lista_populacao_inicial
    populacao_pais = lista_populacao_inicial
    populacao_filhos = lista_populacao_inicial


    fitness_populacao_inicial = np.array([calcula_objetivo_sr(filho, ai, bi, pi, d) for filho in populacao_total])
    populacao_fitness = fitness_populacao_inicial
    n_pop = n_pop_inicial
    taxa_mutacao = taxa_mutacao_inicial/conjunto
    for iter in range(n_iter_ga):

        if usa_coliseu == 1:
            
            inchar = (int(n_pop*2) - len(populacao_total))
            inchar_range = np.arange(0,inchar)
            pop_filhos_temp = np.expand_dims(populacao_total[0],0)
            for _ in inchar_range:
                pop_filhos_temp_temp = np.expand_dims(gerar_filho(populacao_pais, num_pais_duelo, ai, bi, pi, d),0)
                pop_filhos_temp = np.vstack((pop_filhos_temp,pop_filhos_temp_temp))
            populacao_filhos = np.expand_dims(pop_filhos_temp[1:],0)
        
        
        if usa_coliseu == 0:

            pop_filhos_temp = gerar_filho_roleta(populacao_total,populacao_fitness,ai,bi,pi,d)            
            populacao_filhos = np.expand_dims(pop_filhos_temp,0)


        populacao_filhos_mutados = np.expand_dims(pop_filhos_temp[0],0)
        for filho in pop_filhos_temp:
            xmen = np.expand_dims(mutacao(filho, taxa_mutacao, ai, bi, pi, d),0)
            populacao_filhos_mutados = np.vstack((populacao_filhos_mutados,xmen))
        populacao_filhos_mutados = populacao_filhos_mutados[1:]   
        

        populacao_total = np.vstack((populacao_total,populacao_filhos_mutados))
        
        mutantes_fitness = np.array([calcula_objetivo_GA(filho, ai, bi, pi, d)[1] for filho in populacao_filhos_mutados])
        populacao_fitness = np.append(populacao_fitness,mutantes_fitness)
        sobreviventes = oprime_fracos(populacao_fitness, n_pop, taxa_elitismo)




        cromossomos_sobreviventes = populacao_total[sobreviventes==True]
        fitness_sobreviventes = populacao_fitness[sobreviventes==True]

        fitness_sobreviventes_sorted = np.flip(np.argsort(fitness_sobreviventes)[::-1])

        cromossomos_sobreviventes = cromossomos_sobreviventes[fitness_sobreviventes_sorted]
        fitness_sobreviventes = fitness_sobreviventes[fitness_sobreviventes_sorted]


        if fitness_sobreviventes[0] < melhor_obj:
            melhor_obj, melhor_sol = fitness_sobreviventes[0], cromossomos_sobreviventes[0]
            #print(melhor_obj)



        populacao_total = cromossomos_sobreviventes
        populacao_fitness = fitness_sobreviventes
        populacao_pais = populacao_total[:max(2,int(perc_pais_pop*n_pop)+1)]



        if (iter%50==0 or iter==(n_iter_ga-1)):
            #print(conjunto, h, problema, iter, repeticoes)
            guarda_objetivos = np.append(guarda_objetivos,fitness_sobreviventes[0])
        
    return guarda_objetivos[2:],melhor_sol


#testes que serão realizados:
#taxa de mutação = 0.2, 1 , 5
#tipo crossover = 0 , 1
#taxa_elitismo  = 0.35, 0.75, 1
#só filhos mutantes  = 1 , 0

lista_crossover = [0 , 1]
lista_elitismo = [0.35 , 0.75 , 1]
lista_so_filho_mutante = [0 , 1]
lista_taxa_mutacao = [0.2 , 1 , 5]
rodada_numero = 0
numero_rodadas = len(lista_crossover)*len(lista_elitismo)*len(lista_taxa_mutacao)*n_repeticoes*len(conjuntos)*len(lista_hs)*len(lista_problemas)

for cada_tipo_crossover in lista_crossover:
    for cada_elitismo in lista_elitismo:
        for cada_mutacao in lista_taxa_mutacao:
            for repeticoes in range(n_repeticoes):
                for conjunto in conjuntos:
                    for h in lista_hs:
                        for problema in lista_problemas:
                            rodada_numero += 1
                            print("Iniciando rodada {} de {} ({} {} {} {})".format(rodada_numero,numero_rodadas,conjunto, h, problema, repeticoes ))
                            taxa_mutacao_inicial = cada_mutacao
                            taxa_elitismo = cada_elitismo
                            usa_coliseu = cada_tipo_crossover

                            perc_pais_pop = taxa_elitismo





                            inicio=time.time()
                            pi = np.array(dados[conjunto][problema]['pi'])
                            ai = np.array(dados[conjunto][problema]['ai'])
                            bi = np.array(dados[conjunto][problema]['bi'])

                            d=int(sum(pi)*h)

                            lista_populacao_inicial = np.array([solucoes[(c,p,hi,z_corte)] for (c,p,hi,z_corte) in solucoes if conjunto == c and problema == p and h == hi])

                            
                            
                            melhor_obj,melhor_sol = algortimo_genetico(n_iter_ga,taxa_mutacao_inicial,n_pop_inicial,perc_pais_pop,num_pais_duelo,taxa_elitismo,usa_coliseu,ai,bi,pi,d,repeticoes,lista_populacao_inicial,conjunto,h,problema)                
                            
                            fim = time.time()   

                            solucoes_GA[(conjunto, h, problema,taxa_mutacao_inicial, usa_coliseu, taxa_elitismo, repeticoes)] = melhor_sol
                            objetivos_GA[(conjunto, h, problema,taxa_mutacao_inicial, usa_coliseu, taxa_elitismo, repeticoes)] = melhor_obj
                            tempos[(conjunto, h, problema,taxa_mutacao_inicial, usa_coliseu, taxa_elitismo, repeticoes)] = fim-inicio

                




                #breakpoint()
                # for iteracoes in range(1,100):

                # for cromossomo_inicial in lista_populacao_inicial:
                    
                #     for i in range(1, 51):
                #         populacao_total.append(mutacao(cromossomo_inicial, 0.3))
objetivos_pandas = pd.Series(objetivos_GA)
tempos_pandas = pd.Series(tempos)

report = pd.ExcelWriter('resultados_GA.xlsx')

#objetivos_unstack = objetivos_pandas.unstack(level=-3)
#objetivos_unstack.to_excel(report, sheet_name=("Objetivos"))
objetivos_pandas.to_excel(report, sheet_name=("Objetivos"))


#t_unstack = tempos_pandas.unstack(level=-3)
tempos_pandas.to_excel(report,sheet_name="Tempos")

report.save()

with open("solucoes_GA.pkl", "wb") as infile:
    pickle.dump(solucoes_GA, infile)

with open("objetivos_GA.pkl", "wb") as infile:
    pickle.dump(objetivos_GA, infile)

with open("tempos_pandas.pkl", "wb") as infile:
    pickle.dump(tempos, infile)

print("Finalizado em ", (fim-inicio_de_verdade), "segundos")


