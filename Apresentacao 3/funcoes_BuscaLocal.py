from pathlib import Path
import numpy as np
import pandas as pd
from numba import jit
from funcoes_gerais import *

# @jit(nopython=True)
def calcula_objetivo(solucao, ai, bi, pi, d):
    
    set_E, set_T = transforma(solucao)

    if np.sum(pi[set_E])>d:
        return 99999999999
  
    ai_pi = ai[set_E]/pi[set_E] #apenas do set_E
    bi_pi = bi[set_T]/pi[set_T] #apenas do set_T

    ai_pi_decr = np.flip(np.argsort(ai_pi)) #ordem de ai_pi e depois por pi.
    bi_pi_decr = np.flip(np.argsort(bi_pi))
    set_E = set_E[ai_pi_decr]
    set_T = set_T[bi_pi_decr]

    objetivo_minimo = calcula_objetivo_minimo(ai, bi, pi, set_E, set_T, d)

    return objetivo_minimo

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

# @jit(nopython=True)
def _busca_total(solucao_partida, ai, bi, pi, d, lista_tabu): 
    objetivos = {}
    sol_testada = {}
    lista_z = (bi-ai)/(bi+ai)

    for cada_serv in range(0,len(solucao_partida)):
        if cada_serv not in lista_tabu:
            sol_em_teste = solucao_partida.copy()
            sol_em_teste[cada_serv] = not(solucao_partida[cada_serv]) #inverte o serviço de bolso.
            objetivos[cada_serv] = calcula_objetivo(sol_em_teste, ai, bi, pi, d)
            sol_testada[cada_serv] = sol_em_teste

    return objetivos,sol_testada