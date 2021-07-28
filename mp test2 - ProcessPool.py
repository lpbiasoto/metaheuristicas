import concurrent.futures
import time

def soma_ate(numero):
    #print("iniciando soma_ate")
    i = 0 
    while(i<numero):
        i += 1
    #print("calculado ",numero)
    return i

if __name__ == "__main__":
    inicio = time.time()

    somar_esses = [1e7, 2e7, 3e7, 4e7, 5e7, 6e7, 7e7]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(soma_ate,somar_esses)

    fim = time.time()
 
    print(f'Tempo com ProcessPool: {(fim-inicio)}')
    inicio = time.time()
   

    for cada_um in somar_esses:
        soma_ate(cada_um)
        
    fim = time.time()
    print(f'Tempo sem ProcessPool: {(fim-inicio)}')