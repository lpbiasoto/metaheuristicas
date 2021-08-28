from pathlib import Path
import numpy as np
import pandas as pd
from numba import jit
from funcoes_gerais import *

@jit(nopython=True)
def crossover(p1,p2):     #p1,p2 = np.array com soluções, formato booleano.
    tamanho = len(p1)
    metade = int(tamanho/2)
    parte1 = p1[:metade]
    parte2 = p2[metade:]
    filho = np.append(parte1,parte2)
    return filho #falta incluir uma função de reparo de infactíveis

@jit(nopython=True)
def crossover_r(p1,p2):     #p1,p2 = np.array com soluções, formato booleano.
                            #a diferença desse pro anterior é que dois cruzamentos dos mesmos pais
                            #vão retornar filhos diferentes.
                            #a diferença de tempo para 1000 tarefas é de 2us para 20us.
    roleta =  np.array([np.random.random() for _ in range(len(p1))])
    filho = np.copy(p1)
    pai2 = np.copy(p2)
    for cada_tarefa in np.where(roleta<=0.5): #o filho é igual p1, exceto onde roleta<=0.5.
        filho[cada_tarefa] = pai2[cada_tarefa]
        
    return filho #falta incluir uma função de reparo de infactíveis

@jit(nopython=True)
def mutacao(sol,p): #sol = np.array com solução, formato booleano.
                    #p = probabilidade de cada bit sofrer mutação, float ;  [0,1].
    roleta =  np.array([np.random.random() for i in range(len(sol))])
    copia_sol = np.copy(sol)
    for cada_mutante in np.where(roleta<=p):
        copia_sol[cada_mutante] = np.logical_not(copia_sol[cada_mutante])
    return copia_sol

@jit(nopython=True)
def coliseu(sols, n, ai, bi, pi, d):   
    indices_array = len(sols) - 1
    sorteados =  np.array([int(round(np.random.random()*indices_array,0)) for i in range(0,n)]) 
    custo = 999999999999
    for competidor in sols[sorteados]:
        objetivo_competidor = calcula_objetivo_GA(competidor, ai, bi, pi, d)[1]
        if objetivo_competidor < custo:
            custo = objetivo_competidor
            competidor_vencedor = competidor
            objetivo_vencedor = objetivo_competidor
    return competidor_vencedor, objetivo_vencedor

@jit(nopython=True)
def calcula_objetivo_GA(solucao, ai, bi, pi, d):

    set_E, set_T = transforma(solucao)
    
    if np.sum(pi[set_E])>d:
        set_E, set_T = _repara_solucao(set_E,set_T,ai,bi,pi,d)
  
    ai_pi = ai[set_E]/pi[set_E] #apenas do set_E
    bi_pi = bi[set_T]/pi[set_T] #apenas do set_T

    ai_pi_decr = np.flip(np.argsort(ai_pi)) #ordem de ai_pi e depois por pi.
    bi_pi_decr = np.flip(np.argsort(bi_pi))
    set_E = set_E[ai_pi_decr]
    set_T = set_T[bi_pi_decr]

    objetivo_minimo = calcula_objetivo_minimo(ai, bi, pi, set_E, set_T, d)

    return transforma_bin(set_E, set_T), objetivo_minimo

def verifica_factibilidade_e_repara_solucao(solucao, ai, bi, pi, d):
    set_E, set_T = transforma(solucao)
    
    if np.sum(pi[set_E])>d:
        set_E, set_T = _repara_solucao(set_E,set_T,ai,bi,pi,d)
        print(np.sum(pi[set_E])>d)

    solucao_reparada = transforma_bin(set_E, set_T)
    return solucao_reparada

@jit(nopython=True)
def _repara_solucao(set_E,set_T, ai,bi,pi,d):
    sol_ = transforma_bin(set_E,set_T)
    d_solucao = np.sum(pi[set_E])

    if (d_solucao < d): #solução não é infactível, vou sair sem fazer nada.
        return (set_E,set_T)

    z = (bi-ai)/(bi+ai)
    violacao = d_solucao-d #o quanto eu preciso reduzir no meu d_solucao para ser factível?
    zsort_set_E = np.flip(np.argsort(z[set_E])) #os primeiros são os maiores Zs no set_E.
                                                #isto é, supostamente os melhores a serem movidos para o set_T.

    pi_cumsum_zsort_set_E = np.cumsum(pi[zsort_set_E]) #valor acumulado dos pi na ordem considerada a melhor.
    mudar_ate = np.nonzero(pi_cumsum_zsort_set_E>violacao)[0][0] #qual a primeira tarefa da lista ordenada que excede a violação?
    mudar = zsort_set_E[:mudar_ate] #colho os índices dos que precisarão ser alterados
    sol_[mudar] = np.logical_not(sol_[mudar]) #mudo de conjunto itens necessários para reparo
    novo_set_E,novo_set_T = transforma(sol_)

    return novo_set_E,novo_set_T

@jit(nopython=True)
def oprime_fracos(objs,max_pop):
    len_objs = len(objs) 

    sobreviventes = np.array([True]*len_objs) #começo com todos sobrevivendo.
    oprimir_quantos = len_objs - max_pop

    if oprimir_quantos<0: #se eu não tiver exemplares demais, não preciso matar ninguém.
        print(sobreviventes)

    objs_decr = np.flip(np.argsort(objs))

    n_elitismo = max(3,int(max_pop*0.1))
    elite = objs_decr[:n_elitismo]

    roleta =  np.array([np.random.random() for _ in range(len(objs))])
    roleta[elite] = 1 

    oprimidos = np.argsort(roleta)[0:oprimir_quantos]

    sobreviventes[oprimidos] = False
    return sobreviventes

def avaliar_pop(cromossomos, ai, bi, pi, d):
    fitness_cromossomos = []
    for cromossomo in cromossomos:
        fitness_cromossomos.append(calcula_objetivo_GA(cromossomo, ai, bi, pi, d))


@jit(nopython=True)
def calcula_diversidade(sols,objs):
    melhor = np.argmin(objs)
    len_sols = len(sols[0])
    igualdade =np.array([np.sum(sols[kk]==sols[melhor]) for kk in range(len(sols))]) #conta quantos bits iguais a melhor solucao em cada solução
    #de todos bits da população,                                                                           
    #quantos % são diferentes do melhor?
    #substraí len_sols do numerador e denumerador para desconsiderar quando ele comparara a melhor com ela mesmo
    #mas na prática é irrelevante.
    diversidade_percent = (np.sum(igualdade))/((len(igualdade)*len_sols))
    return diversidade_percent