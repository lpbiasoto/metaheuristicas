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
def _repara_solucao(set_E,set_T,ai,bi,pi,d):
    sol_ = transforma_bin(set_E,set_T)
    d_solucao = np.sum(pi[set_E])

    if (d_solucao <= d): #solução não é infactível, vou sair sem fazer nada.
        return (set_E,set_T)

    z = (bi-ai)/(bi+ai)
    violacao = d_solucao-d #o quanto eu preciso reduzir no meu d_solucao para ser factível?
    zsort_set_E = np.flip(np.argsort(z[set_E])) #os primeiros são os maiores Zs no set_E.
                                                #isto é, supostamente os melhores a serem movidos para o set_T.

    pi_cumsum_zsort_set_E = np.cumsum(pi[set_E[zsort_set_E]]) #valor acumulado dos pi na ordem considerada a melhor.
    mudar_ate = np.nonzero(pi_cumsum_zsort_set_E>violacao)[0][0]+1 #qual a primeira tarefa da lista ordenada que excede a violação?
    mudar = set_E[zsort_set_E[:mudar_ate]] #colho os índices dos que precisarão ser alterados
    sol_[mudar] = np.logical_not(sol_[mudar]) #mudo de conjunto itens necessários para reparo
    novo_set_E,novo_set_T = transforma(sol_)
    return novo_set_E,novo_set_T 


@jit(nopython=True)
def repara_solucao(set_E,set_T,ai,bi,pi,d):
    sol_ = transforma_bin(set_E,set_T)
    d_solucao = np.sum(pi[set_E])

    if (d_solucao <= d): #solução não é infactível, vou sair sem fazer nada.
        return (set_E,set_T)

    z = (bi-ai)/(bi+ai)
    violacao = d_solucao-d #o quanto eu preciso reduzir no meu d_solucao para ser factível?
    zsort_set_E = np.flip(np.argsort(z[set_E])) #os primeiros são os maiores Zs no set_E.
                                                #isto é, supostamente os melhores a serem movidos para o set_T.

    pi_cumsum_zsort_set_E = np.cumsum(pi[set_E[zsort_set_E]]) #valor acumulado dos pi na ordem considerada a melhor.
    mudar_ate = np.nonzero(pi_cumsum_zsort_set_E>violacao)[0][0]+1 #qual a primeira tarefa da lista ordenada que excede a violação?
    mudar = set_E[zsort_set_E[:mudar_ate]] #colho os índices dos que precisarão ser alterados
    sol_[mudar] = np.logical_not(sol_[mudar]) #mudo de conjunto itens necessários para reparo
    novo_set_E,novo_set_T = transforma(sol_)
    return novo_set_E,novo_set_T 

@jit(nopython=True)
def oprime_fracos(objs,max_pop,percent_elitismo=0.75):
    len_objs = len(objs) 

    sobreviventes = np.array([True]*len_objs) #começo com todos sobrevivendo.
    oprimir_quantos = len_objs - max_pop

    if oprimir_quantos<=0: #se eu não tiver exemplares demais, não preciso matar ninguém.
        return sobreviventes

    objs_decr = np.argsort(objs)

    n_elitismo = max(3,int(max_pop*percent_elitismo))
    #n_elitismo = 1
    elite = objs_decr[:n_elitismo]

    roleta =  np.array([np.random.random() for i in range(len(objs))])
    roleta[elite] = 1 

    oprimidos = np.argsort(roleta)[0:oprimir_quantos]

    sobreviventes[oprimidos] = False
    return sobreviventes


@jit(nopython=True)
def oprime_fracos_sorte(objs,max_pop):
    len_objs = len(objs) 

    sobreviventes = np.array([True]*len_objs) #começo com todos sobrevivendo.
    oprimir_quantos = len_objs - max_pop

    if oprimir_quantos<=0: #se eu não tiver exemplares demais, não preciso matar ninguém.
        return sobreviventes

    objs_decr = np.argsort(objs)

#    n_elitismo = max(3,int(max_pop*0.1))
    n_elitismo = 1
    elite = objs_decr[:n_elitismo]
    
    max_delta = max(objs)-min(objs)
    delta = objs-min(objs)
    delta_fator = 1-(delta/max_delta)
        

    roleta =  np.array([np.random.random() for i in range(len(objs))])
    roleta[elite] = 1
    
    sorte = roleta+delta_fator #os mais hábeis também tem mais sorte
    
    oprimidos = np.argsort(sorte)[0:oprimir_quantos]

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
    return 1-diversidade_percent


#@jit(nopython=True)
def expande_populacao(sols,objs,taxa_mutacao):
    numero_solucoes = len(sols)
    #número de soluções com paternidade incentivada
    n_melhores = max(1,int(numero_solucoes*0.1)) 
    index_melhores = np.argsort(objs)[-n_melhores:]
    #probabilidade extra reservada para os melhores serem pais.
    incentivo_percent = 0.33
    incentivo_percent_cada = incentivo_percent/n_melhores
    #o restante é distribuído entre todas soluções.
    ampla_disputa_percent = (1-incentivo_percent)
    ampla_disputa_percent_cada = ampla_disputa_percent/numero_solucoes


    #lista com a chance de cada solução ser pai:
    #se fosse uma roleta com circunferência 1,
    #seria o comprimento de arco de cada solução.
    chance_ser_pai = np.array([ampla_disputa_percent_cada]*numero_solucoes)
    chance_ser_pai[index_melhores] += incentivo_percent_cada

    #numero acumulado da chance. 
    #exemplo, se a lista de ticket é [0.3,0.9,1] 
    #e eu sortear 0.5, o ticket sorteado foi o segundo.
    #se eu sortear 0, o ticket sorteado foi o primeiro.
    #se eu sortear 0.90001 foi o terceiro
    ticket = np.cumsum(chance_ser_pai)

    #número de soluções geradas é igual ao número de soluções na entrada.
    #em outras palavras, dobra a população.
    roleta_pai1 = np.array([np.random.random() for nao_importa in range(numero_solucoes)])
    roleta_pai2 = np.array([np.random.random() for nao_importa in range(numero_solucoes)])

    lista_pai1 = np.array([np.sum(cada_sorteio>=ticket) for cada_sorteio in roleta_pai1])
    lista_pai2 = np.array([np.sum(cada_sorteio>=ticket) for cada_sorteio in roleta_pai2])

    filhos = np.array([True]*len(sols[0]))
    for cada_cruzamento in range(numero_solucoes):
        parente1 = sols[lista_pai1[cada_cruzamento]]
        parente2 = sols[lista_pai2[cada_cruzamento]]
        filho_atual = crossover_r(parente1,parente2)
        filhos = np.vstack((filhos,filho_atual))
    filhos = filhos[1:]
    

    mutantes = np.array([True]*len(sols[0]))
    for cada_filho in filhos:
        brucutu = mutacao(cada_filho,taxa_mutacao)
        mutantes = np.vstack((mutantes, brucutu))
        
    mutantes = mutantes[1:]  
    return np.vstack((filhos,mutantes))


def expande_populacao_so_mutantes(sols,objs,taxa_mutacao):
    numero_solucoes = len(sols)    

    mutantes = np.array([True]*len(sols[0]))
    for cada_filho in sols:
        brucutu = mutacao(cada_filho,taxa_mutacao)
        mutantes = np.vstack((mutantes, brucutu))
        
    mutantes2 = np.array([True]*len(sols[0]))
    for cada_filho in sols:
        brucutu = mutacao(cada_filho,taxa_mutacao)
        mutantes2 = np.vstack((mutantes2, brucutu))       
        
    mutantes = mutantes[1:]
    mutantes2 = mutantes2[1:]
    return np.vstack((mutantes,mutantes2))


@jit(nopython=True)
def calcula_objetivo_sr(solucao, ai, bi, pi, d): #sr = sem reparo

    set_E, set_T = transforma(solucao)
    
    if np.sum(pi[set_E])>d:
        return 1e15
  
    ai_pi = ai[set_E]/pi[set_E] #apenas do set_E
    bi_pi = bi[set_T]/pi[set_T] #apenas do set_T

    ai_pi_decr = np.flip(np.argsort(ai_pi)) #ordem de ai_pi e depois por pi.
    bi_pi_decr = np.flip(np.argsort(bi_pi))
    set_E = set_E[ai_pi_decr]
    set_T = set_T[bi_pi_decr]

    objetivo_minimo = calcula_objetivo_minimo(ai, bi, pi, set_E, set_T, d)

    return objetivo_minimo


def avaliar_pop_sr(cromossomos, ai, bi, pi, d):
    fitness_cromossomos = []
    for cromossomo in cromossomos:
        
        fitness_cromossomos.append(calcula_objetivo_sr(cromossomo, ai, bi, pi, d))
    return np.array(fitness_cromossomos)


def exporta_populacao(pop,objs,i): #exporta pop para excel, com objetivos. i é para diferenciar o nome do arquivo e facilitar exportar várias vezes sem fechar o arquivo
    int_pop = np.int_(pop) #transformo os bools em int.
    two_exp_n = np.flip(2**np.arange(len(pop[0]))) #calculo o peso de cada bit
    pop_decimal = np.array([np.sum(int_pop[i]*two_exp_n) for i in range(len(pop))]) #multiplico e somo para saber o valor binário do gene

    #pop_panda = pd.Series(pop_decimal)
    #objs_panda = pd.Series(objs)
    report = pd.ExcelWriter('pop_{}.xlsx'.format(i))
    pop_export_dict = {"genes":pop_decimal,"obj":objs}
    populacao_export = pd.DataFrame(data=pop_export_dict)

    populacao_export.sort_values(by="obj",inplace=True) #ordenar pelos genes para ficar mais tratável

    populacao_export.to_excel(report, sheet_name=("Populacao "+str(i)))
    report.save()

    
    
    