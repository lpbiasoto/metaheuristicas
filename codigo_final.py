import gurobipy as gb
from pathlib import Path
import os
import numpy as np
import pandas as pd
import time

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

dados = {}
objetivos = {}
tempos = {}
dados = extract_data()
lista_hs = [0.8, 0.6, 0.4, 0.2]

conjuntos= [10,20,50,100,200,500,1000]
lista_z = [0.25 , 0.5 , 0.75 , 0.9 , 2]
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
                            break #interrompo a busca. 
                fim = time.time()
                objetivos[(conjunto,problema,h,z_corte)] = seq_obj[-1]
                tempos[(conjunto,problema,h,z_corte)] = fim-inicio
                #print("Conjunto {}, problema {}, h={}, retornou objetivo {}".format(conjunto,problema,h,objetivo))


    fim=time.time()
    #print("Execução em {} segundos".format(fim-inicio))


objetivos_pandas = pd.Series(objetivos)
tempos_pandas = pd.Series(tempos)


report = pd.ExcelWriter('objetivos.xlsx')

for cada_z in lista_z:
    obj = objetivos_pandas[:,:,:,cada_z]
    objetivos_unstack = obj.unstack(level=-2)
    objetivos_unstack.to_excel(report, sheet_name=("z= "+str(cada_z)))

t_unstack = tempos_pandas.unstack(level=-4)
t_unstack.to_excel(report,sheet_name="Tempos")
report.save()