# Universidad de Costa Rica
# Escuela de ingeniería Eléctrica 
# Modelos Probabilísticos de Señales y Sistemas IE-0405
##  Tarea #4 
## Elaborada por María Cordero A. (B42016)

### Primera parte
Para esta sección, se codificó los datos dados en BPSK, es decir que si ingresaba un valor de 1, se colocaba en 0 en la señal de salida, y si se ingresaba un 0, se colocaba un 1. Para transmitir estos datos, se colocó cada bit en una onda portadora sinusoidal, como se indicaba, y se realizó la unión de todos esos datos. Lo anterior se puede verificar en la gráfica indicada como _modulacionBPSK.png_.
![modulacionBPSK.png](attachment:modulacionBPSK.png)

### Segunda parte
Se obtuvo en esta sección la potencia promedio a partir de la potencia instantánea de la onda obtenida anteriormente. Esta última se obtuvo elevando al cuadrado la onda obtenida, ya que los datos eran finitos y discretos. Seguidamente, se obtuvo la potencia promedio conociendo que ésta se encuentra de la forma según Sadiku y Alexander (2006):<br>
                $P_{prom}= \frac{1}{T}\int_{t}^{t+T} P_{inst} dt$ <br>
De ahí, se obtuvo que para esta señal, se tiene un valor de 0.49.

### Tercera parte
A continuación, se creó un canal de ruido AWGN, el cual es utilizado comúnmente para evaluar telecomunicaciones. Para esto, se especificó distintos valores de SNR, que consiste en el radio de potencia de la señal útil que tendrá este ruido, y se utiliza para caracterizarlo (Gharaibeh, 2012). Para esto, se utilizó 6 valores desde -2 hasta 3, obteniendo así un ruido diferente para cada valor de SNR. Seguidamente se graficó cada uno de resultados para verificar las variaciones entre cada uno, como se puede apreciar en las gráficas de nombre _AWGN#_. Se pudo observar que se mantenía la forma de onda original muy sutilmente, pero había distorsión en los valores. También se notó que, a medida que aumentaba el valor del SNR, la distorsión era mayor, ya que indica que habría menos potencia "útil", por lo que se va perdiendo la forma de onda.

### Cuarta parte
En esta sección, se obtuvo la densidad espectral de la potencia, la cual es una caracterización del ruido de una señal que permite, mediante el espectro de ésta, conocer su origen, a partir de la potencia por unidad de frecuencia (Boreman, 1999). Se observa en las gráficas de nombre _DE#_ que el espectro es similar al concatenar la señal original con el ruido, por lo que se puede asumir que esto significa que provienen de una misma fuente. Asimismo, se ve que son diferentes al espectro en frecuencia de la señal original. 

### Quinta parte
Se deseaba obtener la tasa de error de cada señal luego de ingresar el ruido, por lo que se hizo mediante el análisis de la energía de llegada, cuantizándola. Esto significa que, si se recibió un valor en alto (mayor de la mitad), se decodificaba a 0, ya que se trata de una codificación BPSK. De igual forma, si se recibió un valor en bajo, se decodificaba a 1. A partir de ahí, se contó el número de errores al encontrar la diferencia entre los valores de salida y de llegada, lo cual se sumaba finalmente para obtener el número de errores. Para obtener el BER, se dividió esta cantidad entre el número de bits original, obteniendo lo siguiente:<br>

"Hay 20.0 errores en los 10000 bits recibidos con un SNR = -2 para una tasa de error de 0.002. <br>
Hay 19.0 errores en los 10000 bits recibidos con un SNR = -1 para una tasa de error de 0.0019.<br>
Hay 15.0 errores en los 10000 bits recibidos con un SNR = 0 para una tasa de error de 0.0015.<br>
Hay 14.0 errores en los 10000 bits recibidos con un SNR = 1 para una tasa de error de 0.0014.<br>
Hay 13.0 errores en los 10000 bits recibidos con un SNR = 2 para una tasa de error de 0.0013.<br>
Hay 18.0 errores en los 10000 bits recibidos con un SNR = 3 para una tasa de error de 0.0018."<br>
Como se puede observar, a mayor magnitud del SNR, mayor tasa de error, sin embargo, los errores son suficientemente pequeños para decir que la señal no ha llegado correctamente, por lo que se puede decir que la decodificación es exitosa.

### Sexta Parte
Como se observa en la gráfica denominada _BERvsSNR.png_, los valores del BER definitivamente varían con el SNR de la señal de ruido, por lo que puede pensar que BER es dependiente del SNR.

## Bibliografía
- Boreman, Glenn.(1999)._Fundamentos de electro-óptica para ingenieros_. USA: Optical Engineering Press.
- Gharaibeh, Khaled. (2012)._Nonlinear Distortion in Wireless Systems: Modeling and Simulation with MATLAB_. Sussex: Wiley.
- Sadiku, M. & Alexander, C.(2006) _Fundamentos de circuitos electrónicos_. México: McGraw-Hill.
