import multiprocessing as mp
import time

n_processos = 24 
#ref:  Python Multiprocessing Tutorial: Run Code in Parallel Using the Multiprocessing Module
#Corey Schafer
#https://www.youtube.com/watch?v=fKl2JW_qrso

def sum_while():
    print("iniciando sum_while()")
    start = time.time()
    now = time.time()
    i = 0 
    while(now-start<1):
        i += 1
        now = time.time()
    print(i)

if __name__ == '__main__':
#iniciando sem paralelismo
    inicio = time.perf_counter()

    for _ in range(n_processos):    
        sum_while()
    fim = time.perf_counter()

    print(f'Tempo sem paralelismo: {fim-inicio}')

# iniciando com paralelismo

    inicio = time.perf_counter()

    processos = []
    for _ in range(n_processos):
        p = mp.Process(target=sum_while)
        p.start()
        processos.append(p)

    for cada_processo in processos:
        cada_processo.join()


    fim = time.perf_counter()

    print(f'Tempo com paralelismo: {fim-inicio}')


#conclusão mais óbvia:
#isso me permite fazer muito mais calculos, 
# (A soma dos valores retornados é muito maior usando mp para cada segundo executando o problema)
# 
#conclusão menos óbvia:
# mas como aqui meu critério é tempo, eu fico menos tempo em cada execução, 
#principalmente quando o número de processos é maior que o número de CPUs.
#(a soma de cada execução é menor, mas eu retornei várias em um segundo).
 