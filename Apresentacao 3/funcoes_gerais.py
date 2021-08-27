from pathlib import Path
import math
import os
import numpy as np
import pandas as pd
import time
from numba import jit
import pickle
from statistics import mean

def compara_objetivos_biskup(objetivos_min):
    biskup = pd.read_excel('objetivos_biskup.xlsx').set_index(["Unnamed: 0","Unnamed: 1", "Unnamed: 2"])
    objetivos_biskup = biskup.to_dict()["objetivo"]
    resultados = {}

    for key in objetivos_biskup:
        resultados[key] = (objetivos_min[key] - objetivos_biskup[key])/objetivos_biskup[key]
    
    return resultados, mean(resultados[k] for k in resultados)

def compara_objetivos_entre_etapas(objetivos_antes, objetivos_depois):
    resultados = {}

    for key in objetivos_antes:
        resultados[key] = (objetivos_depois[key] - objetivos_antes[key])/objetivos_antes[key]
    
    return resultados, mean(resultados[k] for k in resultados)

@jit(nopython=True)
def transforma(solucao):
    set_E = np.nonzero(solucao == True)[0] #[0] para retornar o np array e não o tuple.
    set_T = np.nonzero(solucao == False)[0]
    return set_E,set_T

@jit(nopython=True)
def transforma_bin(set_E, set_T):
    solucao_bin = np.array([True]*(len(set_E)+len(set_T)))
    solucao_bin[set_T] = False
    return solucao_bin

@jit(nopython=True)
def calcula_objetivo_minimo(ai, bi, pi, set_E, set_T, d):
    
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