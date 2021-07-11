from pathlib import Path
import os
import numpy as np
import pandas as pd

raiz = Path.cwd() #capturo diretório atual
pasta_dados = Path.joinpath(raiz,"dados") #defino onde estão os problemas
problemas = os.listdir(pasta_dados) #lista os arquivos na pasta dos problemas
prob_por_arquivo = 10 #informando quantos problemas por arquivo eu tenho
for conjunto in problemas:  #circulo nos arquivos
    print("Iniciando problema {}".format(conjunto))
    caminho_problema = Path.joinpath(pasta_dados,conjunto)
    df = pd.read_fwf(caminho_problema) #importo no df
    df.columns=["n","p","a","b"]
    df.drop(columns=["n"], inplace=True) #jogo fora columa inútil
    n=df.iloc[0,0] #guardo o tamanho de itens do problema
    print("n = ", n)  
    df.dropna(inplace=True) #jogo fora linhas inúteis
    for problema_numero in range(prob_por_arquivo): #vou circular nos problemas dentro do arquivo. Cada iteração deste for é um problema.
        prob = df[n*problema_numero:n*(problema_numero+1)]
        pi = prob["p"].to_numpy()
        ai = prob["a"].to_numpy()
        bi = prob["b"].to_numpy()

        ## nesse ponto tenho o parâmetros do problema atual guardadas nas variáveis pi, ai e bi, prontos para serem utilizados na modelagem.
   