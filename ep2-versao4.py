# Importação das bibliotecas
import matplotlib.pyplot as plt
import numpy as np
import random
import math




# Definicao das forçantes fk a partir do enunciado da parte 2
def f_c(p,t,x,dx):

  h=dx # Definicao de h
  r=(10*(1+ np.cos(5*(t)))) #r(t)

  # Condicao das forcantes pontuais
  if p-h/2<=x and x<=p+h/2:
    ghx=1/h
  else:
    ghx=0

  fk=r*ghx
  return fk



# Tarefa 2, Sub-Tarefa A (utilizado no EP2 sem grandes modificações, função auxiliar ao método de Crank-Nicolson que manipula matrizes)
def fatoraçãoLDLt(N, lamb, b):

    # Criação da matriz A (já em vetores subdiagonal e diagonal)
    diag = []
    subdiag = []
    for m in range (N-1):
        diag.append(1+lamb)
    for k in range (N-2):
        subdiag.append(-lamb/2)

    # Define valor n
    tamanho = (len(diag))

    # Constrói matrizes L e D vazias
    D = []
    L = []

    # Definição da decomposição LDLt feita no EP1
    D.append(diag[0])
    for i in range(tamanho-1):
        L.append(subdiag[i]/D[i])
        D.append(diag[i+1] - D[i]*L[i]*L[i])

 
    # Passos de sistemas lineares (como já especificado no EP1) => Ax=b -> LZ=b + DY=Z + L_tx=Y
    tamanho2 = len(b)
    y = []
    y.append(b[0])
    for i in range (1,tamanho2):
        y.append(b[i] - L[i-1]*y[i-1])

    z = []
    for i in range (tamanho2):
        z.append(y[i]/D[i])

    x = []
    x.append(z[tamanho2-1])
    j = 0
    for i in range (tamanho2-2, -1, -1):
        x.append(z[i] - L[i]*x[j])
        j = j+1
    x.reverse()
    return x




# Tarefa 2C do EP1 (Método de Crank-Nicolson) com pequenas modificações
def tarefa_2c_adapt(N, M, p):

    # Definição de variáveis a partir de seus conceitos no enunciado
    lamb = N #Devido M = N
    T = 1
    dt = float(1/N)
            
    #Condição inicial
    U = []  #armazena os valores do vetor x do sistema Ax=b 
    b = []  #armazena os valores do vetor b do sistema Ax=b

    for i in range (1,N):
        b.append(0)

    U = b.copy()
    b.clear()

    # Iterações do nosso método de Crank-Nicolson
    for k in range(0,M):

        # Definição do tk e tk+1
        t = (k+1)*dt
        t_anterior = k*dt
        
        for i in range(1,N):
            x = i*dt  # Definição do xi

            # Calculo da função fk em um tempo tk e no tempo tk+1 (via fórmula 35 do enunciado)
            f = f_c(p, t, x, dt)
            f_anterior = f_c(p, t_anterior, x, dt)
        
            # Equacionamento da fórmula (35) do enunciado
            if i == 1:
                b.append(U[i-1] + (lamb/2)*(-2*U[i-1] + U[i]) + (dt/2)*(f + f_anterior))
            elif i == N-1:
                b.append(U[i-1] + (lamb/2)*(U[i-2] - 2*U[i-1]) + (dt/2)*(f + f_anterior))
            else:
                b.append(U[i-1] + (lamb/2)*(U[i-2] - 2*U[i-1] + U[i]) + (dt/2)*(f + f_anterior))

        # Decomposição a partir da matriz A (LDLt) e a descoberta do x (de Ax=b via sistemas lineares)
        resposta = fatoraçãoLDLt(N, lamb, b)

        b.clear()
        U.clear()
        U = resposta.copy()
    
    # No EP2, só retornamos a uk(T, xi)
    return U




# Nova decomposição LDLt para (desta vez) uma matriz A não esparsa
def LDL_decomp(A):

  # Define valor n
  tamanho = (len(A))

  # Constrói matrizes L e D vazias
  D = []
  L = []
  for i in range (tamanho):
      for j in range (tamanho):
          d  = [0] * tamanho
          l = [0] * tamanho
      D.append(d)
      L.append(l)

  # ---------- Passos referenciados ao livro teórico (cap 6.6 Burden/Faires)
  
  # Passo 1
  for i in range (tamanho): 

      soma0 = 0
      v = []

      # Passo 2
      for j in range (i): 
          e = float(L[i][j]) * float (D[j][j])
          v.append(e)
          soma0 = soma0 + float(L[i][j]) * float(L[i][j]) * float (D[j][j])

      # Passo 3
      D[i][i] = A[i][i] - soma0

      # Passo 4
      for j in range (i, tamanho):
          soma1 = 0 
          for k in range (i):
              soma1 = soma1 + float(L[j][k]) * float (v[k])
          L[j][i] = ((A[j][i] - soma1)/(float(D[i][i])))
  
  # Retornando em formato LDLt
  matrizes3 = [L, D]
  return matrizes3
  




# Montar matriz e definir calculo de intensidades
def sistema_normal(M, N, nf, uT):

  # Calculando cada uk para um pnf específico
  vetorNF = []
  for k in range (len(nf)):
    vetorNF.append(tarefa_2c_adapt(N, M, nf[k]))

  
  # Criando a matriz dos produtos internos <uk, uk> para cada k de 1 a nf (chamado de a)
  # Criando o vetor dos produtos internos <uT, uk> para cada k de 1 a nf (chamado de b)
  A = []
  B = []
  for i in range (len(nf)):
    m = []
    B.append(np.dot(uT, vetorNF[i]))
    for j in range (len(nf)):
        a = np.dot(vetorNF[i], vetorNF[j])
        m.append(a)
    A.append(m)
  

  # Encontrando x em LDLt*x = B
  Adecomposta = LDL_decomp(A)

  # Manipulação de sistema linear similar a ideia do EP1 (porém com algumas modificações)
  # Ax=b -> LZ=b + DY=Z + L_tx=Y

  L = Adecomposta[0]
  D = Adecomposta[1]

  Z=[]
  Z.append(B[0])
  for i in range (1,len(B)):
      soma0 = 0
      for j in range (i):
          soma0 = soma0 + float(Z[j]) * float (L[i][j])

      z_novo = B[i] - soma0
      Z.append(z_novo)

  Y=[]
  for j in range (len(Z)):
      y=float(Z[j])/float(D[j][j])
      Y.append(y)

  X=[]
  for a in range (len(Y)):
      X.append(0)

  a=len(Y)-1
  X[a]=Y[a]
  for i in range (1,len(Y)):

      soma1 = 0
      for j in range (len(Y), (a-i), -1):
          soma1 = soma1 + float(X[j-1]) * float (L[j-1][a-i])

      X[a-i] = Y[a-i] - soma1

  return X




# Leitura de arquivo teste.txt original
def ler_arquivo(arquivo, prox_posicao):

  uT = []
  with open(arquivo) as arq:
      
      linha0 = arq.readline() # Ignoramos a primeira linha que é os pnf
      
      linha = arq.readline()
      contador = 0
      while linha:

        # Selecionamos o xi ao depender do N escolhido
        if (contador % prox_posicao == 0):
            a = float(linha.strip())
            uT.append(a)

        linha = arq.readline()
        contador += 1

  arq.close()
  
  # Remoção dos valores de fronteiras
  uT.pop(-1)
  uT.pop(0)

  return uT




# Implementação do ruído conforme o enunciado descreve para o item D
def ruido(uT):
  uT_novo = []

  for i in range (len(uT)):

    # Produção de um valor novo para cada uT contendo o erro randômico solicitado
    m = float(uT[i]) * (1 + 0.01 * ((random.random() - 0.5) * 2))
    uT_novo.append(m)
  
  return uT_novo




# Definição da fórmula (39) do enunciado do EP2 (utilizado para calculo de erro nos itens C e D)
def erroMMQ (resposta, uT, N, nf):

  M = N
  soma1 = 0
  UkT = []

  for k in range (1, len(nf)+1):
    UkT.append(tarefa_2c_adapt(N, M, nf[k-1]))

  for i in range (N-1):
    soma0 = 0

    for k in range (len(nf)):
      soma0 = soma0 + float(resposta[k]) * float (UkT[k][i])
    soma1 = soma1 + (uT[i] - soma0)**2
  
  erro2 = (((1/N)*soma1)**(1/2))
  return erro2



# Recalculagem do uT (o novo após a descoberta das intensidades)
def recalc(resposta, N, nf):
    UkT = []
    M = N
    uT_novo = np.zeros(N-1)
    for k in range (len(nf)):
        UkT.append(tarefa_2c_adapt(N, M, nf[k]))
        uT_novo += resposta[k] * np.array(UkT[k])
       
    return uT_novo



# Plotagem de gráficos para a solução em T=1
def plotagem(N, gabarito, nome1, resposta, nome2):
  x=[]
  uT_novo = np.zeros(N-1)
  for a in range (1,N):
    x.append(a/N)


  plt.plot(x, gabarito, label=nome1)
  plt.plot(x, resposta, label=nome2)
  plt.ylabel('Temperatura na Barra')
  plt.xlabel('Posição na Barra')
  plt.legend()
  plt.show()



# Interface do programa
def main():

  loopInterface = 1
  while (loopInterface == 1):

    print('\nOs testes requisitados para o relatório são:\nA , B , C , D (digite uma dessas letras em CapsLock):')
    teste = input()
    if teste =='A':
      print('\n\nO teste selecionado foi o teste A \nOs parametros são N = 128, nf = 1 e p1 = 0.35. Aqui o uT(xi) = 7*u1(T, xi).\n')

      # Ajustando valores para o item A
      nf = [0.35]
      N = 128
      M = N

      # Definindo função uT(xi)
      x = tarefa_2c_adapt(N, M, nf[0])
      uT = np.array(x) * 7

      # Imprimindo intensidades encontradas
      resposta = sistema_normal(M, N, nf, uT)
      print('\nO vetor coluna das intensidades encontradas foi: \n ')
      print(resposta)
      print('\nPortanto, a1 = ' + str(round(resposta[0], 3)) + '\n\n')

      # Calculo do erro e impressão
      erro = erroMMQ(resposta, uT, N, nf)
      print('\nO erro encontrado para este item foi:')
      print(erro)
        

    elif teste =='B':
      print('\n\nO teste selecionado foi o teste B \nOs parametros são N = 128, nf = 4 e p1 = 0.15, p2 = 0.3, p3 = 0.7 e p4 = 0.8. \nAqui o uT(xi) = 2.3*u1(T, xi) + 3.7*u2(T, xi) + 0.3*u3(T, xi) + 4.2*u4(T, xi).\n')

      # Ajustando valores para o item B
      nf = [0.15, 0.3, 0.7, 0.8]
      N = 128
      M = N

      # Definindo função uT(xi)
      x = tarefa_2c_adapt(N, M, nf[0])
      y = tarefa_2c_adapt(N, M, nf[1])
      z = tarefa_2c_adapt(N, M, nf[2])
      w = tarefa_2c_adapt(N, M, nf[3])
      uT = np.array(x) * 2.3 + np.array(y) * 3.7 + np.array(z) * 0.3 + np.array(w) * 4.2

      # Imprimindo intensidades encontradas
      resposta = sistema_normal(M, N, nf, uT)
      print('\nO vetor coluna das intensidades encontradas foi: \n ')
      print(resposta)
      print('\nPortanto, os valores das intensidades em 3 casas decimais são: a1 = ' + str(round(resposta[0], 3)) + ', a2 = ' + str(round(resposta[1], 3)) + ', a3 = ' + str(round(resposta[2], 3)) + ', a4 = ' + str(round(resposta[3], 3)) +'\n\n')

      # Calculo do erro e impressão
      erro = erroMMQ(resposta, uT, N, nf)
      print('\nO erro encontrado para este item foi:')
      print(erro)

    elif teste =='C':

      print('\n\nO teste selecionado foi o teste C. \nAqui o uT(xi) foi fornecido no arquivo teste.txt da disciplina\n')
      print('Digite um valor de N válido (os valores são 128, 256, 512, 1024, 2048):')
      
      # Ajustando valores para o item C
      nf = [0.14999999999999999, 0.20000000000000001, 0.29999999999999999, 0.34999999999999998, 0.5, 0.59999999999999998, 0.69999999999999996, 0.72999999999999998, 0.84999999999999998, 0.90000000000000002]
      N = int(input())
      M = N

      # Leitura de arquivo teste.txt original
      prox_posicao = 2048/N

      print ('\nInsira o caminho para o arquivo fornecido pela disciplina (denominado teste.txt)')
      arquivo = str(input()) # Para meu PC caminho é D:/Users/André/Desktop/MAP_EP2/teste.txt

      uT = ler_arquivo(arquivo, prox_posicao)


      # Imprimindo intensidades encontradas
      resposta = sistema_normal(M, N, nf, uT)
      print('\nO vetor coluna das intensidades encontradas foi: \n ')
      print(resposta)
      
      print('\nPortanto, os valores das intensidades em 9 casas decimais são:\n\na1 = ' + str(round(resposta[0], 9)) + ', a2 = ' + str(round(resposta[1], 9)) + ', a3 = ' + str(round(resposta[2], 9)) + ', a4 = ' + str(round(resposta[3], 9)) + ', a5 = ' + str(round(resposta[4], 9)))
      print('a6 = ' + str(round(resposta[5], 9)) + ', a7 = ' + str(round(resposta[6], 9)) + ', a8 = ' + str(round(resposta[7], 9)) + ', a9 = ' + str(round(resposta[8], 9)) + ', a10 = ' + str(round(resposta[9], 9))+ '\n\n')


      # Calculo do erro e impressão
      erro = erroMMQ(resposta, uT, N, nf)
      print('\nO erro encontrado para este item foi:')
      print(erro)
 
      #Plotagem dos gráficos
      uT_novo = recalc(resposta, N, nf)
      plotagem(N, uT, 'uT lido no arquivo', uT_novo , 'uT novo')

    elif teste =='D':

      print ('\n\nO teste selecionado foi o teste D. \nAqui o uT(xi) foi fornecido no arquivo teste.txt da disciplina\n')
      print ('Este teste é similar ao feito no item C, porém, acrescido da problematica de ruídos.')
      print ('Digite um valor de N válido (os valores são 128, 256, 512, 1024, 2048):')

      # Ajustando valores para o item D
      nf = [0.14999999999999999, 0.20000000000000001, 0.29999999999999999, 0.34999999999999998, 0.5, 0.59999999999999998, 0.69999999999999996, 0.72999999999999998, 0.84999999999999998, 0.90000000000000002]
      N = int(input())
      M = N

      # Leitura de arquivo teste.txt original
      prox_posicao = 2048/N

      print ('\nInsira o caminho para o arquivo fornecido pela disciplina (denominado teste.txt)')
      arquivo = str(input()) # Para meu PC caminho é 'D:/Users/André/Desktop/MAP_EP2/teste.txt'

      uT = ler_arquivo(arquivo, prox_posicao)

      # Inserção do ruído
      uT = ruido(uT)

      # Imprimindo intensidades encontradas
      resposta = sistema_normal(M, N, nf, uT)
      print('\nO vetor coluna das intensidades encontradas foi: \n ')
      print(resposta)
      
      print('\nPortanto, os valores das intensidades em 9 casas decimais são:\n\na1 = ' + str(round(resposta[0], 9)) + ', a2 = ' + str(round(resposta[1], 9)) + ', a3 = ' + str(round(resposta[2], 9)) + ', a4 = ' + str(round(resposta[3], 9)) + ', a5 = ' + str(round(resposta[4], 9)))
      print('a6 = ' + str(round(resposta[5], 9)) + ', a7 = ' + str(round(resposta[6], 9)) + ', a8 = ' + str(round(resposta[7], 9)) + ', a9 = ' + str(round(resposta[8], 9)) + ', a10 = ' + str(round(resposta[9], 9))+ '\n\n')


      # Calculo do erro e impressão
      erro = erroMMQ(resposta, uT, N, nf)
      print('\nO erro encontrado para este item foi:')
      print(erro)

      #Plotagem dos gráficos
      uT_novo = recalc(resposta, N, nf)
      plotagem(N, uT, 'uT lido no arquivo após ruido', uT_novo , 'uT novo')

    # Loop do programa para fazer outros testes
    print('\nDeseja fazer outro teste no programa? (s/n)')
    finalizar = input()
    if (finalizar == "n"):
      loopInterface = 0

main()