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
def transforma_hc(solucao):
    set_E = np.nonzero(solucao == True)[0] #[0] para retornar o np array e não o tuple.
    set_T = np.nonzero(solucao == False)[0]
    return set_E,set_T

def transforma_bin_hc(set_E, set_T, conjunto):
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
            objetivos[cada_serv] = calcula_objetivo_sr(sol_em_teste, ai, bi, pi, d)
            sol_testada[cada_serv] = sol_em_teste

    return objetivos,sol_testada

def busca_local(set_E, set_T, conjunto, ai, bi, pi, d, seq_obj):
    solucao_pre_buscalocal = transforma_bin_hc(set_E, set_T, conjunto)
                
    objs = {}
    sols = {}
    seq_sols = []
    seq_sols.append(solucao_pre_buscalocal)
    # obj_original = seq_obj[-1]
    obj_atual = seq_obj[-1]
    # continua = True
    # contador_aleatorio = 0
    lista_tabu = [conjunto+1]
    # contador_tabu = 0
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
                if contador_pior >= max(conjunto*0.1, 10): #2º critério de parada - número máximo de soluções piores aceitas  
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
solucao_busca = {}
tempos_construtiva = {}
tempos = {}
dados = extract_data()

with open("dados.pkl", "wb") as infile:
    pickle.dump(dados, infile)


lista_hs = [0.8]# , 0.6, 0.4, 0.2]
conjuntos = [10]#,20,50,100,200,500,1000]

lista_z = [0.25]# , 0.5 , 0.6 , 0.75, 2]
#lista_problemas = list(range(1,11))
lista_problemas = [1]

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
                sol_bin = transforma_bin(set_E,set_T)
                obj_inicial_rebobinada = calcula_objetivo(sol_bin, ai, bi , pi , d)
                z_corte_E = (bi[set_E]- ai[set_E] / (bi[set_E]+ai[set_E]))
                zsort_set_E = (np.argsort(z_corte_E)) #Coloquei os maiores primeiro
                                    #significa as tarefas com "piores zcortes para o set E" primeiro
                
                sol_atual_rebobinar = np.copy(sol_bin)
                objetivo_atual_rebobinada = obj_inicial_rebobinada
                #vou passar um por um, até o objetivo parar de melhorar.
                #"rebobinando os itens de volta pro set T."
                #print("{} - itens no set_E - iniciando".format(sum(sol_atual_rebobinar)))
                for cada_adiantado in set_E[zsort_set_E]: 
                    #print(cada_adiantado, ai[cada_adiantado],bi[cada_adiantado],pi[cada_adiantado])
                    sol_nova = np.copy(sol_atual_rebobinar)

                    sol_nova[cada_adiantado] = False
                    #print("{} - itens no set_E".format(sum(sol_nova)))
                    obj_novo = calcula_objetivo(sol_nova, ai, bi , pi , d)

                    if obj_novo <= objetivo_atual_rebobinada:
                        sol_atual_rebobinar = np.copy(sol_nova)
                        objetivo_atual_rebobinada = obj_novo
                    else:
                        break
                

               
                set_E,set_T = transforma(sol_atual_rebobinar)

                seq_obj, seq_sols = busca_local(set_E, set_T, conjunto, ai, bi, pi, d, seq_obj)

                ####### FIM DA BUSCA LOCAL #######
                    
                
                ####### CONSOLIDANDO OS RESULTADOS #######
                print(conjunto,problema,h,z_corte)
                objetivos[(conjunto,problema,h,z_corte)] = seq_obj[-1]
                solucao_busca[(conjunto,problema,h,z_corte)] = seq_sols[-1]
                if (conjunto,problema,h) in objetivos_min:
                    if seq_obj[-1] < objetivos_min[(conjunto,problema,h)]:
                        objetivos_min[(conjunto,problema,h)] = seq_obj[-1]
                        solucoes_min[(conjunto,problema,h)] = transforma_hc(seq_sols[-1])
                else:
                    objetivos_min[(conjunto,problema,h)] = seq_obj[-1]
                    solucoes_min[(conjunto,problema,h)] = transforma_hc(seq_sols[-1]) 
                fim = time.time()
                    
                tempos[(conjunto,problema,h,z_corte)] = fim-meio
                tempos_construtiva[(conjunto,problema,h,z_corte)] = meio-inicio

    fim=time.time()

objetivos_pandas = pd.Series(objetivos_min)
tempos_pandas = pd.Series(tempos)
tempos_pandas2 = pd.Series(tempos_construtiva)
solucao_busca_pandas = pd.Series(solucao_busca)
solucao_busca_pandas.to_pickle("solucoes.pkl")

with open("objetivos_min.pkl", "wb") as infile:
    pickle.dump(objetivos_min, infile)


##### INICIO ALGORITMO GENETICO #####
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

conjuntos = [10,20,50,100,200,500,1000]

lista_z = [0.25 , 0.5 , 0.6 , 0.75, 2]
lista_problemas = list(range(1,11))


populacao = {}
qtd_pop_inicial = {}
n_pop_inicial = 1000 #max(50,conjunto*0.1)
n_iter_ga = 1000
n_repeticoes = 4 
num_pais_duelo = 2

taxa_mutacao_inicial = 1
taxa_elitismo = 1
usa_coliseu = 0
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


rodada_numero = 0
numero_rodadas = n_repeticoes*len(conjuntos)*len(lista_hs)*len(lista_problemas)

for repeticoes in range(n_repeticoes):
    for conjunto in conjuntos:
        for h in lista_hs:
            for problema in lista_problemas:
                rodada_numero += 1
                print("Iniciando rodada {} de {} ({} {} {} {})".format(rodada_numero,numero_rodadas,conjunto, h, problema, repeticoes ))

                inicio=time.time()
                pi = np.array(dados[conjunto][problema]['pi'])
                ai = np.array(dados[conjunto][problema]['ai'])
                bi = np.array(dados[conjunto][problema]['bi'])

                d=int(sum(pi)*h)

                lista_populacao_inicial = np.array([solucoes[(c,p,hi,z_corte)] for (c,p,hi,z_corte) in solucoes if conjunto == c and problema == p and h == hi])

                
                
                melhor_obj,melhor_sol = algortimo_genetico(n_iter_ga,taxa_mutacao_inicial,n_pop_inicial,perc_pais_pop,num_pais_duelo,taxa_elitismo,usa_coliseu,ai,bi,pi,d,repeticoes,lista_populacao_inicial,conjunto,h,problema)                
                
                fim = time.time()   

                solucoes_GA[(conjunto, h, problema, repeticoes)] = melhor_sol
                objetivos_GA[(conjunto, h, problema, repeticoes)] = melhor_obj
                tempos[(conjunto, h, problema, repeticoes)] = fim-inicio    

objetivos_pandas = pd.Series(objetivos_GA)
tempos_pandas = pd.Series(tempos)

report = pd.ExcelWriter('resultados_GA.xlsx')

objetivos_pandas.to_excel(report, sheet_name=("Objetivos"))


tempos_pandas.to_excel(report,sheet_name="Tempos")

report.save()

with open("solucoes_GA.pkl", "wb") as infile:
    pickle.dump(solucoes_GA, infile)

with open("objetivos_GA.pkl", "wb") as infile:
    pickle.dump(objetivos_GA, infile)

with open("tempos_pandas.pkl", "wb") as infile:
    pickle.dump(tempos, infile)


print("Finalizado em ", (fim-inicio_de_verdade), "segundos")


