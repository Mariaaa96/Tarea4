#!/usr/bin/env python
# coding: utf-8

# # Universidad de Costa Rica
# # Escuela de ingeniería Eléctrica 
# # Modelos Probabilísticos de Señales y Sistemas IE-0405
# ##  Tarea #4 
# ## Elaborada por María Cordero A. (B42016)

# ### Primera Parte
# Crear un esquema de modulación BPSK para los bits presentados. Esto implica asignar una forma de onda sinusoidal normalizada (amplitud unitaria) para cada bit y luego una concatenación de todas estas formas de onda.

# In[101]:


import pandas as pd
import os
import math
import numpy as np 
import csv
from scipy.stats import norm
import matplotlib.mlab as mlab
import scipy
from scipy import stats 
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import signal

#Se inicializa una lista
lista = []

#Se llena la lista con los valores del csv
with open('bits10k.csv', newline='') as archivo:
	lectura = csv.reader(archivo)
	for fila in lectura:
		lista.append(fila)     
        
#Nueva lista con integers - Método largo, complicado y probablemente innecesario, pero funciona y lo ocupo
listaF=[]

for i in range(N):
    if lista[i] == ['0']:
        listaF.append(0)
    if lista[i] == ['1']:
        listaF.append(1)

        
N=len(lista)  

#Se codifica los datos en BPSK
datos=[]

for i in range(N):
    if lista[i] == ['0']:
        datos.append(1)
    if lista[i] == ['1']:
        datos.append(0)
        
bitNum = 10000    
#Frecuencia dada
f=5000

#Duración del período de cada onda
T=1/f

#Número de puntos de muestreo 
sample=50

#Vector de tiempo 
tiempo = np.linspace (0, T, sample)

#Creación de la forma de onda
coseno = np.cos(2*np.pi*f*tiempo)

#Frecuencia de muestreo 
fs=sample/T

#Tiempo para toda la señal
t=np.linspace(0,bitNum*T,bitNum*sample)

#Se crea un Vector con N*P ceros
vecFinal = np.zeros(bitNum*sample)

#Creación de la señal BPSK
for k,b in enumerate(datos):
    vecFinal[k*sample:(k+1)*sample] = (2*b-1)*coseno
      

#Visualización de los primeros bit modulados
pb = 5
plt.figure()
plt.plot(vecFinal[0:pb*sample])
plt.title('Modulación BPSK')
plt.xlabel('Tiempo (s)')
plt.savefig('modulacionBPSK.png')


# ### Segunda parte:
# Calcular la potencia promedio de la señal modulada generada.

# In[102]:


#Potencia instantánea de la señal modulada
Pinst= vecFinal**2

#Potencia promedio mediante la integral de la potencia instantánea entre el período
Ps=integrate.trapz(Pinst,t)/(T*bitNum)
print('La potencia promedio de la señal modulada es de ', Ps)


# ### Tercera parte:
#  Simular un canal ruidoso del tipo AWGN (ruido aditivo blanco gaussiano) con una relación señal a ruido (SNR) desde -2 hasta 3 dB.

# In[103]:


#Crear ruido AWGN para SNR=-2
SNR1 = -2
# Funciones de potencia, desviación, ruido y simulación de canal
def potencia (valueSNR):
    return Ps / (10**(SNR1 / 10))

def standarDeviation (pRuido):
    return np.sqrt(pRuido)

def ruido (ds):
    return np.random.normal(0, ds, vecFinal.shape)

def canalRuidoso(vectorRuidoso):
    return vecFinal+ vectorRuidoso

#Para SNR =-2
Pn_2=potencia(-2)
sigma_2 = standarDeviation(Pn_2)
ruido_2 = ruido(sigma_2)
Rx_2=canalRuidoso(ruido_2)
plt.figure()
plt.plot(Rx_2[0:pb*sample])
plt.title("Simulación de la señal original con un canal ruidoso tipo AWGN con SNR de -2")
plt.savefig('AWGN_2.png')
plt.show()

#Para SNR =-1
Pn_1=potencia(-1)
sigma_1 = standarDeviation(Pn_1)
ruido_1 = ruido(sigma_1)
Rx_1=canalRuidoso(ruido_1)
plt.figure()
plt.plot(Rx_1[0:pb*sample])
plt.title("Simulación de la señal original con un canal ruidoso tipo AWGN con SNR de -1")
plt.savefig('AWGN_1.png')
plt.show()

#Para SNR =0
Pn0=potencia(0)
sigma0 = standarDeviation(Pn0)
ruido0 = ruido(sigma0)
Rx0=canalRuidoso(ruido0)
plt.figure()
plt.plot(Rx0[0:pb*sample])
plt.title("Simulación de la señal original con un canal ruidoso tipo AWGN con SNR de 0")
plt.savefig('AWGN0.png')
plt.show()

#Para SNR =1
Pn1=potencia(1)
sigma1 = standarDeviation(Pn1)
ruido1 = ruido(sigma1)
Rx1=canalRuidoso(ruido1)
plt.figure()
plt.plot(Rx1[0:pb*sample])
plt.title("Simulación de la señal original con un canal ruidoso tipo AWGN con SNR de 1")
plt.savefig('AWGN1.png')
plt.show()

#Para SNR =2
Pn2=potencia(2)
sigma2 = standarDeviation(Pn2)
ruido2 = ruido(sigma2)
Rx2=canalRuidoso(ruido2)
plt.figure()
plt.plot(Rx2[0:pb*sample])
plt.title("Simulación de la señal original con un canal ruidoso tipo AWGN con SNR de 2")
plt.savefig('AWGN2.png')
plt.show()

#Para SNR =3
Pn3=potencia(3)
sigma3 = standarDeviation(Pn3)
ruido3 = ruido(sigma3)
Rx3=canalRuidoso(ruido3)
plt.figure()
plt.plot(Rx3[0:pb*sample])
plt.title("Simulación de la señal original con un canal ruidoso tipo AWGN con SNR de 3")
plt.savefig('AWGN3.png')
plt.show()


# ### Cuarta parte
#  Graficar la densidad espectral de potencia de la señal con el método de Welch (SciPy), antes y después del canal ruidoso.

# In[104]:


fw, Pxx=signal.welch(vecFinal, fs, nperseg=1024)
plt.semilogy(fw,Pxx)
plt.title('Densidad espectral de potencia antes del canal ruidoso')
plt.savefig('DEsAWGN.png')
plt.figure()

#SNR-2
f_2, P_2=signal.welch(Rx_2, fs, nperseg=1024)
plt.semilogy(f_2,P_2)
plt.title('Densidad espectral de potencia después del canal ruidoso SNR =-2')
plt.savefig('DE_2.png')
plt.figure()

#SNR-1
f_1, P_1=signal.welch(Rx_1, fs, nperseg=1024)
plt.semilogy(f_1,P_1)
plt.title('Densidad espectral de potencia después del canal ruidoso SNR =-1')
plt.savefig('DE_1.png')
plt.figure()

#SNR0
f0, P0=signal.welch(Rx0, fs, nperseg=1024)
plt.semilogy(f0,P0)
plt.title('Densidad espectral de potencia después del canal ruidoso SNR =0')
plt.savefig('DE0.png')
plt.figure()

#SNR1
f1, P1=signal.welch(Rx1, fs, nperseg=1024)
plt.semilogy(f1,P1)
plt.title('Densidad espectral de potencia después del canal ruidoso SNR =1')
plt.savefig('DE1.png')
plt.figure()

#SNR2
f2, P2=signal.welch(Rx2, fs, nperseg=1024)
plt.semilogy(f2,P2)
plt.title('Densidad espectral de potencia después del canal ruidoso SNR =2')
plt.savefig('DE2.png')
plt.figure()

#SNR3
f3, P3=signal.welch(Rx3, fs, nperseg=1024)
plt.semilogy(f3,P3)
plt.title('Densidad espectral de potencia después del canal ruidoso SNR =3')
plt.savefig('DE3.png')
plt.figure()


# ### Quinta parte
# Demodular y decodificar la señal y hacer un conteo de la tasa de error de bits (BER, bit error rate) para cada nivel SNR.

# In[105]:


# Energia que presenta la onda (discreta)
energia = np.sum(coseno**2)
    #Se pasa la lista creada anteriormente a un arreglo porque así me permite trabajar al decodificar
arreglo=np.array(listaF)

#Se define la función del error y del BER
def errorDecod(bitsRecibidos):
    return np.sum(np.abs(arreglo - bitsRecibidos))

def BER (err):
    return err/bitNum

#SNR-2
    # Bits recibidos al decodificar
recibidos_2 = np.zeros(arreglo.shape)
    # Decodificación de la señal por detección de energía
for k, b in enumerate(arreglo):
    Ep = np.sum(Rx_2[k*sample:(k+1)*sample] * coseno)
    if Ep > Es/2:
        recibidos_2[k] = 0
    else:
        recibidos_2[k] = 1

error_2= errorDecod(recibidos_2)
BER_2=BER(error_2)

print('Hay {} errores en los {} bits recibidos con un SNR = -2 para una tasa de error de {}.'.format(error_2, bitNum, BER_2))

#SNR-1
    # Bits recibidos al decodificar
recibidos_1 = np.zeros(arreglo.shape)
    # Decodificación de la señal por detección de energía
for k, b in enumerate(arreglo):
    Ep = np.sum(Rx_1[k*sample:(k+1)*sample] * coseno)
    if Ep > Es/2:
        recibidos_1[k] = 0
    else:
        recibidos_1[k] = 1

error_1= errorDecod(recibidos_1)
BER_1=BER(error_1)

print('Hay {} errores en los {} bits recibidos con un SNR = -1 para una tasa de error de {}.'.format(error_1, bitNum, BER_1))

#SNR0
    # Bits recibidos al decodificar
recibidos0 = np.zeros(arreglo.shape)
    # Decodificación de la señal por detección de energía
for k, b in enumerate(arreglo):
    Ep = np.sum(Rx0[k*sample:(k+1)*sample] * coseno)
    if Ep > Es/2:
        recibidos0[k] = 0
    else:
        recibidos0[k] = 1

error0= errorDecod(recibidos0)
BER0=BER(error0)

print('Hay {} errores en los {} bits recibidos con un SNR = 0 para una tasa de error de {}.'.format(error0, bitNum, BER0))


#SNR1
    # Bits recibidos al decodificar
recibidos1 = np.zeros(arreglo.shape)
    # Decodificación de la señal por detección de energía
for k, b in enumerate(arreglo):
    Ep = np.sum(Rx1[k*sample:(k+1)*sample] * coseno)
    if Ep > Es/2:
        recibidos1[k] = 0
    else:
        recibidos1[k] = 1

error1= errorDecod(recibidos1)
BER1=BER(error1)

print('Hay {} errores en los {} bits recibidos con un SNR = 1 para una tasa de error de {}.'.format(error1, bitNum, BER1))

#SNR2
    # Bits recibidos al decodificar
recibidos2 = np.zeros(arreglo.shape)
    # Decodificación de la señal por detección de energía
for k, b in enumerate(arreglo):
    Ep = np.sum(Rx2[k*sample:(k+1)*sample] * coseno)
    if Ep > Es/2:
        recibidos2[k] = 0
    else:
        recibidos2[k] = 1

error2= errorDecod(recibidos2)
BER2=BER(error2)

print('Hay {} errores en los {} bits recibidos con un SNR = 2 para una tasa de error de {}.'.format(error2, bitNum, BER2))

#SNR3
    # Bits recibidos al decodificar
recibidos3 = np.zeros(arreglo.shape)
    # Decodificación de la señal por detección de energía
for k, b in enumerate(arreglo):
    Ep = np.sum(Rx3[k*sample:(k+1)*sample] * coseno)
    if Ep > Es/2:
        recibidos3[k] = 0
    else:
        recibidos3[k] = 1

error3= errorDecod(recibidos3)
BER3=BER(error3)

print('Hay {} errores en los {} bits recibidos con un SNR = 3 para una tasa de error de {}.'.format(error3, bitNum, BER3))


# ### Sexta parte
# Graficar BER versus SNR.

# In[106]:


SNR=np.linspace(-2, 3,6) 
print(SNR)
BER=[BER_2,BER_1,BER0, BER1,BER2, BER3]

plt.figure()
plt.plot(SNR, BER)
plt.title("Gráfica de BER en función de SNR")
plt.xlabel('SNR')
plt.ylabel('BER')
plt.savefig('BERvsSNR.png')
plt.show()


# In[ ]:




