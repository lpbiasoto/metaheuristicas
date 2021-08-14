import gurobipy as gb
from pathlib import Path
import math
import os
import numpy as np
import pandas as pd
import time
from numba import jit

def extract_data():
    raiz = Path.cwd() #capturo diretório atual
    pasta_dados = Path.joinpath(raiz,"dados") #defino onde estão os problemas
    prob_por_arquivo = 10 #informando quantos problemas por arquivo eu tenho
    conjuntos = [10,20,50,100,200,500,1000]

    for conjunto in conjuntos:  #circulo nos arquivos
        dados[conjunto] = {}
        arquivo = f"sch"+str(conjunto)+'.txt'
        print("Iniciando problema {}".format(arquivo))
        caminho_problema = Path.joinpath(pasta_dados,arquivo)
        df = pd.read_fwf(caminho_problema) #importo no df
        df.columns=["n","p","a","b"]
        df.drop(columns=["n"], inplace=True) #jogo fora columa inútil
        n=df.iloc[0,0] #guardo o tamanho de itens do problema
        print("n = ", n)  
        df.dropna(inplace=True) #jogo fora linhas inúteis
        for problema_numero in range(prob_por_arquivo): #vou circular nos problemas dentro do arquivo. Cada iteração deste for é um problema.
            prob = df[n*problema_numero:n*(problema_numero+1)]
            dados[conjunto][problema_numero+1] = {
                'pi': prob["p"].to_numpy(),
                'ai': prob["a"].to_numpy(),
                'bi': prob["b"].to_numpy()
            }
            ## nesse ponto tenho o parâmetros do problema atual guardadas nas variáveis pi, ai e bi, prontos para serem utilizados na modelagem.
    return dados

@jit(nopython=True)
def transforma(solucao):
    set_E = np.nonzero(solucao == True)[0] #[0] para retornar o np array e não o tuple.
    set_T = np.nonzero(solucao == False)[0]
    return set_E,set_T

def transforma_bin(set_E, set_T, conjunto):
    solucao = range(0, conjunto)
    solucao_bin = np.isin(solucao, set_E)
    return solucao_bin

# @jit(nopython=True)
def busca_total(solucao_partida, ai, bi, pi, d, lista_tabu): 
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

@jit(nopython=True)
def maximo(a,b):
    if a>b:
        return a
    else:
        return b
    

@jit(nopython=True)
def minimo(a,b):
    if a<b:
        return a
    else:
        return b

@jit(nopython=True)
def calcula_objetivo(solucao, ai, bi, pi, d):
    
    set_E, set_T = transforma(solucao)
    
    if np.sum(pi[set_E])>d:
        return 99999999999

        
    ai_pi = ai[set_E]/pi[set_E] #apenas do set_E
    bi_pi = bi[set_T]/pi[set_T] #apenas do set_T
    set_E_pi = pi[set_E]    
    
    set_T_pi = pi[set_T]
    ai_pi_decr = np.flip(np.argsort(ai_pi)) #ordem de ai_pi e depois por pi.
    bi_pi_decr = np.flip(np.argsort(bi_pi))
    set_E = set_E[ai_pi_decr]
    set_T = set_T[bi_pi_decr]

    objetivo_minimo = calcula_objetivo_minimo(ai, bi, pi, set_E, set_T, d)

    # breakpoint()
    return objetivo_minimo

@jit(nopython=True)
def verifica_factibilidade(solucao, pi, d):
    set_E, set_T = transforma(solucao)
        
    if np.sum(pi[set_E])>d:
        return False
    else:
        return True

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

    return minimo(objetivo_final_d,objetivo_inicio_0)

@jit(nopython=True)
def calcula_objetivo_0(ai, bi, pi, set_E, set_T):
    sum_pi = 0
    objetivo = 0

    gap_E = d - np.sum(pi[set_E])
    sum_pi = gap_E

    for tarefa in set_E:
        objetivo += sum_pi * ai[tarefa]
        sum_pi += pi[tarefa]
    
    sum_pi = -gap_E
    for tarefa in set_T:
        objetivo += (sum_pi+pi[tarefa])*bi[tarefa]
        sum_pi += pi[tarefa]
    return objetivo

def busca_local(set_E, set_T, conjunto, ai, bi, pi, d, seq_obj):
    solucao_pre_buscalocal = transforma_bin(set_E, set_T, conjunto)
                
    objs = {}
    sols = {}
    seq_sols = []
    seq_sols.append(solucao_pre_buscalocal)
    obj_original = seq_obj[-1]
    obj_atual = seq_obj[-1]
    continua = True
    contador_aleatorio = 0
    lista_tabu = [conjunto+1]
    contador_tabu = 0
    aceita_pior = False
    contador_pior = 0

    for n in range(1,101): #1º critério de parada - número máximo de vizinhanças                    
        obj_ant = obj_atual
        if n == 1:
            objs,sols = busca_total(solucao_pre_buscalocal, ai, bi, pi, d, lista_tabu)
        else:
            objs,sols = busca_total(sol_atual, ai, bi, pi, d, lista_tabu)

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
                if contador_pior >= maximo(conjunto*0.1, 10): #2º critério de parada - número máximo de soluções piores aceitas  
                    break
                else:
                    contador_pior += 1
        elif obj_atual < seq_obj[-1]:
            seq_obj.append(obj_atual)
            seq_sols.append(sol_atual)
            contador_pior = 0
    return seq_obj, seq_sols



dados = {}
objetivos = {}
objetivos_min = {}
solucoes_min = {}
tempos_construtiva = {}
tempos = {}
dados = extract_data()
lista_hs = [0.8, 0.6, 0.4, 0.2]
conjuntos= [10,20,50,100,200,500,1000]
lista_z = [0.25 , 0.5 , 0.75 , 0.9, 2]
lista_problemas = list(range(1,11))


for conjunto in conjuntos:
    for z_corte in lista_z:
        for problema in lista_problemas:
            for h in lista_hs:
                inicio=time.time()
                pi = np.array(dados[conjunto][problema]['pi'])
                ai = np.array(dados[conjunto][problema]['ai'])
                bi = np.array(dados[conjunto][problema]['bi'])

                d=int(sum(pi)*h)

                i_itens = len(pi)
                i_index=list(range(i_itens))
                i_np = np.array(i_index)

                set_T = [] #Tardiness set
                set_E = [] #Earliness set
                E_pi = [] #lista dos tempo de serviço alocados em E
                E_tr = d #Tempo restante no conjunto E


                ####### INÍCIO DA HEURÍSTICA CONSTRUTIVA #######

                #pelo princípio do V-shaped, só preciso tomar a decisão de quais produtos ficam no set_E e quais no set_T
                #vou sugerir uma forma de decidir quais vão pro set_E:

                z= (bi-ai)/(ai+bi) 
                #Z é valor entre -1 e 1 referente a vantagem de se colocar depois.
                # Valores positivos indicam vantagem de inserir antes, valores negativos de inserir depois.
                # outra opção seria tentar a z/pi, de forma a tentar preencher o d com a melhor eficiência possível.
                #falta imnplementar uma maneira de organizar por um segundo melhor indicador caso haja empate.

                z_sort = np.flip(np.argsort(z)) #sequência decrescente. Primeiros itens da lista = preferência por set_E.

                candidatos_E =  np.array([z_sort[z[z_sort]>z_corte]][0])
                #z[z_sort] retorna os valores de Z ordenados por z_sort.
                #z[z_sort]>z_corte retorna True onde os z[z_sort] é maior que z_corte
                #[z_sort[z[z_sort]>z_corte]] retorna uma lista, na ordem de z_sort, onde z é maior que z_corte

                lista_dos_que_cabem = np.nonzero(np.cumsum(pi[candidatos_E])<d)
                set_E = i_np[z_sort][lista_dos_que_cabem]

                set_T = np.array([z_sort[z[z_sort]<-z_corte]][0]) #guarda no set_T itens com preferencia pelo set_T



                #o que não foi alocado preferencialmente em nenhum set
                #deverá ser alocado usando outra heurística.
                #quais itens restaram?
                #pensando na lista ordenada por z_sort,
                #tiro um pedaço do começo e outro do fim.
                #a do começo tem tamanho:
                alocado_em_E = len(set_E)
                #do fim tem tamanho:
                alocado_em_T = len(set_T) 

                #i_itens é o tamanho da minha lista.

                sem_pref = i_np[z_sort][alocado_em_E:(i_itens-alocado_em_T)]
                ai_sp = ai[sem_pref]
                bi_sp = bi[sem_pref]
                pi_sp = pi[sem_pref]

                ai_pi = ai_sp/pi_sp
                bi_pi = bi_sp/pi_sp


                ai_pi_decr = np.flip(np.argsort(ai_pi)) #índices dos componentes com menores ai/pi
                bi_pi_decr = np.flip(np.argsort(bi_pi)) #índices dos componentes com menores bi/pi

                d_subgrupo = d - np.sum(pi[set_E])
                E_tr = d_subgrupo #Tempo restante no conjunto E
                sub_E = []
                sub_T = []

                max_iter = conjunto
                indice_candidato = 0  #quando buscar
                for i in range(max_iter):

                    candidato = bi_pi_decr[indice_candidato] #maior bi/pi não alocado é o candidato
                    if E_tr >= pi_sp[candidato]: #se o candidato couber no set E, fazer:
                        sub_E.append(candidato) #insere ele no subset E
                        #print("Incluindo candidato {} no grupo E".format(candidato))
                        ai_pi_decr = np.delete(ai_pi_decr, np.where(ai_pi_decr==candidato)) #retira da lista de candidatos ai 
                        bi_pi_decr = np.delete(bi_pi_decr, np.where(bi_pi_decr==candidato)) #retira da lista de candidatos bi

                        if len(bi_pi_decr) == 0:
                            break
                        sub_T.append(ai_pi_decr[0]) #insere no sub T
                        #print("Incluindo candidato {} no grupo T".format(ai_pi_decr[0]))
                        ai_pi_decr = np.delete(ai_pi_decr, np.where(ai_pi_decr==sub_T[-1])) #retira da lista de candidatos 
                        bi_pi_decr = np.delete(bi_pi_decr, np.where(bi_pi_decr==sub_T[-1])) #retira da lista candidatos bi

                        E_tr = d_subgrupo - np.sum(pi_sp[sub_E])
                    else:
                        bi_pi_decr = np.delete(bi_pi_decr, np.where(pi_sp[bi_pi_decr]>E_tr))
                    if len(bi_pi_decr) == 0:
                        break

                for i in ai_pi_decr:
                    sub_T.append(i)
                    #print("Incluindo candidato {} no grupo T".format(sub_T[-1]))

                #traduzindo indices do subgrupo "sem prioridade" nos sets principais.
                sub_E_traduzido = sem_pref[sub_E]
                sub_T_traduzido = sem_pref[sub_T]            
                set_E = np.append(set_E,sub_E_traduzido)
                set_T = np.append(set_T,sub_T_traduzido)

                ordem_E = np.argsort(ai[set_E]/pi[set_E])
                ordem_T = np.argsort(bi[set_T]/pi[set_T])    

                set_T = np.flip(set_T[ordem_T]) #reordenando os sets.
                set_E =  np.flip(set_E[ordem_E]) #set_E é invertido.
                #o primeiro item da lista termina em "d"


                si={}
                si[set_E[0]] = d - pi[set_E[0]] #o primeiro termina no d.




                for i in range(1,len(set_E)):
                    si[set_E[i]] = si[set_E[i-1]] - pi[set_E[i]]

                si[set_T[0]] = d
                for i in range(1,len(set_T)):
                    si[set_T[i]] = si[set_T[i-1]] + pi[set_T[i-1]]

                E = {}
                T = {}
                for i in i_index:
                    E[i] = max(d-si[i]-pi[i],0)
                    T[i] = max(pi[i]+si[i]-d,0)   

                objetivo = []
                for i in i_index:
                    objetivo.append(T[i]*bi[i])
                    objetivo.append(E[i]*ai[i])
                seq_obj = [sum(objetivo)]
                
                
                
                #iniciando etapa de busca.
                
                for ii in range(len(set_T)):
                    E_tr = d-np.sum(pi[set_E])
                    T_ordem_z = np.argsort(z[set_T])
                    set_Tz = np.flip(set_T[T_ordem_z])

                    candidato = set_Tz[0]
                    if pi[candidato]>E_tr:
                        lista_dos_que_cabem = np.nonzero(pi[set_Tz]<=E_tr)
                        if len(lista_dos_que_cabem)>0:
                            break #não cabe mais nenhum
                        candidato=lista_dos_que_cabem[0] #escolho o primeiro que cabe.

                    if pi[candidato]<=E_tr: #continuando com um candidato que cabe.
                        set_E_ant = set_E.copy()
                        set_T_ant = set_T.copy()
                        
                        set_E = np.append(set_E,candidato)
                        set_T = np.delete(set_T,np.where(set_T==candidato))

                        ordem_E = np.argsort(ai[set_E]/pi[set_E])    
                        set_E = np.flip(set_E[ordem_E]) #reordenando os sets.
                        si={}
                        si[set_E[0]] = d - pi[set_E[0]] #o primeiro termina no d.

                        for i in range(1,len(set_E)):
                            si[set_E[i]] = si[set_E[i-1]] - pi[set_E[i]]

                        si[set_T[0]] = d
                        for i in range(1,len(set_T)):
                            si[set_T[i]] = si[set_T[i-1]] + pi[set_T[i-1]]

                        E = {}
                        T = {}
                        for i in i_index:
                            E[i] = max(d-si[i]-pi[i],0)
                            T[i] = max(pi[i]+si[i]-d,0)   

                        objetivo = []
                        for i in i_index:
                            objetivo.append(T[i]*bi[i])
                            objetivo.append(E[i]*ai[i])

                        seq_obj.append(sum(objetivo))
                        if seq_obj[-1]>seq_obj[-2]:
                            del seq_obj[-1] #garantindo que a melhor solução está na última posição. 
                            set_E = set_E_ant.copy()
                            set_T = set_T_ant.copy()
                            break #interrompo a busca. 

                ####### FIM DA HEURÍSTICA CONSTRUTIVA #######

                meio = time.time()
                
                

                seq_obj, seq_sols = busca_local(set_E, set_T, conjunto, ai, bi, pi, d, seq_obj)

                ####### FIM DA BUSCA LOCAL #######
                    
                
                ####### CONSOLIDANDO OS RESULTADOS #######
                print(conjunto,problema,h,z_corte)
                objetivos[(conjunto,problema,h,z_corte)] = seq_obj[-1]
                if (conjunto,problema,h) in objetivos_min:
                    if seq_obj[-1] < objetivos_min[(conjunto,problema,h)]:
                        objetivos_min[(conjunto,problema,h)] = seq_obj[-1]
                        solucoes_min[(conjunto,problema,h)] = transforma(seq_sols[-1])
                else:
                    objetivos_min[(conjunto,problema,h)] = seq_obj[-1]
                    solucoes_min[(conjunto,problema,h)] = transforma(seq_sols[-1]) 
                fim = time.time()
                    
                tempos[(conjunto,problema,h,z_corte)] = fim-meio
                tempos_construtiva[(conjunto,problema,h,z_corte)] = meio-inicio

    fim=time.time()

objetivos_pandas = pd.Series(objetivos)
tempos_pandas = pd.Series(tempos)
tempos_pandas2 = pd.Series(tempos_construtiva)


report = pd.ExcelWriter('resultados.xlsx')

for cada_z in lista_z:
    obj = objetivos_pandas[:,:,:,cada_z]
    objetivos_unstack = obj.unstack(level=-2)
    objetivos_unstack.to_excel(report, sheet_name=("z= "+str(cada_z)))
t_unstack = tempos_pandas.unstack(level=-4)
t_unstack.to_excel(report,sheet_name="Tempos")

t_unstack = tempos_pandas2.unstack(level=-4)
t_unstack.to_excel(report,sheet_name="Tempos_construtiva")
report.save()