from pathlib import Path
import numpy as np
import pandas as pd
from numba import jit

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
    roleta =  np.array([np.random.random() for i in range(len(p1))])
    filho = np.copy(p1)
    p2 = np.copy(p2)
    for cada_tarefa in np.where(roleta<=0.5): #o filho é igual p1, exceto onde roleta<=0.5.
        filho[cada_tarefa] = p2[cada_tarefa]
        
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
def _calcula_objetivo(solucao, ai, bi, pi, d):
    
    set_E, set_T = transforma(solucao)
    
    if np.sum(pi[set_E])>d:
        return 99999999999
  
    ai_pi = ai[set_E]/pi[set_E] #apenas do set_E
    bi_pi = bi[set_T]/pi[set_T] #apenas do set_T

    ai_pi_decr = np.flip(np.argsort(ai_pi)) #ordem de ai_pi e depois por pi.
    bi_pi_decr = np.flip(np.argsort(bi_pi))
    set_E = set_E[ai_pi_decr]
    set_T = set_T[bi_pi_decr]

    objetivo_minimo = _calcula_objetivo_minimo(ai, bi, pi, set_E, set_T, d)

    return objetivo_minimo

@jit(nopython=True)
def coliseu(sols, n, ai, bi, pi, d):   
    indices_array = len(sols) - 1
    sorteados =  np.array([int(round(np.random.random()*indices_array,0)) for i in range(0,n)]) 
    custo = 999999999999
    for competidor in sols[sorteados]:
        
        objetivo_competidor = _calcula_objetivo_GA(competidor, ai, bi, pi, d)
        if objetivo_competidor < custo:
            competidor_vencedor = competidor
            objetivo_vencedor = objetivo_competidor
    return competidor_vencedor, objetivo_vencedor

@jit(nopython=True)
def _calcula_objetivo_GA(solucao, ai, bi, pi, d):
    
    set_E, set_T = transforma(solucao)
    
    if np.sum(pi[set_E])>d:
        return 99999999999
  
    ai_pi = ai[set_E]/pi[set_E] #apenas do set_E
    bi_pi = bi[set_T]/pi[set_T] #apenas do set_T

    ai_pi_decr = np.flip(np.argsort(ai_pi)) #ordem de ai_pi e depois por pi.
    bi_pi_decr = np.flip(np.argsort(bi_pi))
    set_E = set_E[ai_pi_decr]
    set_T = set_T[bi_pi_decr]

    objetivo_minimo = _calcula_objetivo_minimo(ai, bi, pi, set_E, set_T, d)

    return objetivo_minimo


@jit(nopython=True)
def _calcula_objetivo_minimo(ai, bi, pi, set_E, set_T, d):
    
    sum_pi_final_d = 0
    objetivo_final_d = 0
    objetivo_inicio_0 = 0

    gap_E = d - np.sum(pi[set_E])
    sum_pi_inicio_0 = gap_E
    
    for tarefa in set_E:
        objetivo_final_d += sum_pi_final_d * ai[tarefa]
        objetivo_inicio_0 += sum_pi_inicio_0 * ai[tarefa]
        sum_pi_final_d += pi[tarefa]
        sum_pi_inicio_0 += pi[tarefa]
    
    sum_pi_final_d = 0
    sum_pi_inicio_0 = -gap_E
    for tarefa in set_T:
        objetivo_final_d += (sum_pi_final_d+pi[tarefa])*bi[tarefa]
        objetivo_inicio_0 += (sum_pi_inicio_0+pi[tarefa])*bi[tarefa]
        sum_pi_final_d += pi[tarefa]
        sum_pi_inicio_0 += pi[tarefa]

    if gap_E > pi[set_T[0]]:
        objetivo_inicio_0 = 99999999999

    return min(objetivo_final_d,objetivo_inicio_0)

def busca_local(set_E, set_T, conjunto, ai, bi, pi, d, seq_obj):
    solucao_pre_buscalocal = transforma_bin(set_E, conjunto)
                
    objs = {}
    sols = {}
    seq_sols = []
    seq_sols.append(solucao_pre_buscalocal)
    obj_atual = seq_obj[-1]
    lista_tabu = [conjunto+1]
    aceita_pior = False
    contador_pior = 0

    for n in range(1,101): #1º critério de parada - número máximo de vizinhanças                    
        obj_ant = obj_atual
        if n == 1:
            objs,sols = _busca_total(solucao_pre_buscalocal, ai, bi, pi, d, lista_tabu)
        else:
            objs,sols = _busca_total(sol_atual, ai, bi, pi, d, lista_tabu)

        objs = dict(sorted(objs.items(), key=lambda item: item[1], reverse=False))
        if len(objs) == 0:
            break
        obj_atual = objs[next(iter(objs))]
        sol_atual = sols[next(iter(objs))]
        if obj_atual >= obj_ant:
            if aceita_pior == False:
                aceita_pior = True
                contador_pior = 1
            else:
                if contador_pior >= max(conjunto*0.1, 10): #2º critério de parada - número máximo de soluções piores aceitas  
                    break
                else:
                    contador_pior += 1
        elif obj_atual < seq_obj[-1]:
            seq_obj.append(obj_atual)
            seq_sols.append(sol_atual)
            contador_pior = 0
    return seq_obj, seq_sols


@jit(nopython=True)
def transforma(solucao):
    set_E = np.nonzero(solucao == True)[0] #[0] para retornar o np array e não o tuple.
    set_T = np.nonzero(solucao == False)[0]
    return set_E,set_T

def transforma_bin(set_E, conjunto):
    solucao = range(0, conjunto)
    solucao_bin = np.isin(solucao, set_E)
    return solucao_bin

# @jit(nopython=True)
def _busca_total(solucao_partida, ai, bi, pi, d, lista_tabu): 
    objetivos = {}
    sol_testada = {}
    lista_z = (bi-ai)/(bi+ai)

    for cada_serv in range(0,len(solucao_partida)):
        if cada_serv not in lista_tabu:
            sol_em_teste = solucao_partida.copy()
            sol_em_teste[cada_serv] = not(solucao_partida[cada_serv]) #inverte o serviço de bolso.
            objetivos[cada_serv] = _calcula_objetivo(sol_em_teste, ai, bi, pi, d)
            sol_testada[cada_serv] = sol_em_teste

    return objetivos,sol_testada