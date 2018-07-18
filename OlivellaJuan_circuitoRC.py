import numpy as np
import matplotlib.pylab as plt


# Descarga los datos del archivo CircuitoRC.txt

datos = np.genfromtxt("CircuitoRC.txt")

t = datos[:,0]  # Datos del tiempo

q = datos[:,1] # Datos de carga

iteraciones = 5000 # PARA MAYOR PRECISION AUMENTE EL VALOR DE LAS ITERACIONES


# Define la funcion de maxima verosimilitud

def verosimilitud(qobservado, qmodelo):

    chicuadrado = sum((qobservado-qmodelo)**2)/10000
    
    return (np.exp(chicuadrado*(-0.5)))


# Define la funcion que establece mi modelo de la carga acumulada en un condensador

def carga(tobservada, R, C):
    
    return (10*C)*(1-np.exp(-(tobservada)/(R*C)))


# Define el camino que se debe seguir para encontrar el R que minimiza el error del modelo

caminoR = np.empty((0))

Rguess = np.random.random()*(20)

caminoR = np.append(caminoR, Rguess)


# Define el camino que se debe seguir para encontrar el C que minimiza el error del modelo

caminoC = np.empty((0))

Cguess = np.random.random()*(20)

caminoC = np.append(caminoC, Cguess)


# Define el camino que se debe seguir para encontrar la funcion de verosimilitud que minimiza el error del modelo

caminoVer = np.empty((0))

Verguess = verosimilitud(q, carga(t, Rguess, Cguess))

caminoVer = np.append(caminoVer, Verguess)


# Define la caminata que se debe seguir para acercarce a los valores de R y C que minimizan la funcion

for i in range(iteraciones):
    
    Rfuturo = np.random.normal(caminoR[i], 0.2) # MODIFIQUE LA DESVIACION PARA MAYOR PRECISION
    Cfuturo = np.random.normal(caminoC[i], 0.2)
    Verfuturo =verosimilitud(q, carga(t, Rfuturo, Cfuturo))
    
    alpha = Verfuturo/verosimilitud(q, carga(t, caminoR[i],caminoC[i]))

    if(alpha >= 1.0):
        caminoR  = np.append(caminoR,Rfuturo)
        caminoC  = np.append(caminoC,Cfuturo)
        caminoVer = np.append(caminoVer,Verfuturo)
                
    else:
        beta = np.random.random()
        
        if(beta <= alpha):
            caminoR = np.append(caminoR,Rfuturo)
            caminoC = np.append(caminoC,Cfuturo)
            caminoVer = np.append(caminoVer,Verfuturo)
            
        else:
            caminoR = np.append(caminoR,caminoR[i])
            caminoC = np.append(caminoC,caminoC[i])
            caminoVer = np.append(caminoVer,caminoVer[i])

# Devuelve la posicion de la maxima verosimilitud que optimiza el modelo. Del mismo modo el valor de R y C para el modelo

maxVer = np.argmax(caminoVer)

maxR = np.amax(caminoR[maxVer])

maxC = np.amax(caminoC[maxVer])


# Grafica los datos originales y el modelo estimado. Los valores optimos de R y C se imprimen en la pantalla

print("El valor de la resistencia y de la capacitancia para el modelo son respectivamente:", maxR, maxC)


fig, ax = plt.subplots()
plt.title("Carga del condensador en funcion del tiempo")
ax.scatter(t,q, s = 2, label ="Observado")
ax.plot(t, carga(t, maxR, maxC), c = "r",label ="Modelo")
plt.legend(loc = 0)
plt.xlabel("$t$")
plt.ylabel("$q(t)$")
plt.savefig("CargaRC.pdf")


# Graficas de R y C en funcion de la funcion de verosimilitud y susrespectivos histogramas

fig = plt.figure()

plt.subplot(2,2,1)
plt.scatter(caminoR, -np.log(caminoVer), s = 2)
plt.title("-Log(L) vs. R")
plt.xlabel("Camino de R")
plt.ylabel("-Log(L)")


plt.subplot(2,2,2)
plt.scatter(caminoC, -np.log(caminoVer), s = 2)
plt.title("-Log(L) vs. C")
plt.xlabel("Camino de C")
plt.ylabel("-Log(L)")

plt.subplot(2,2,3)
plt.hist(caminoR)
plt.title("Histograma de R")
plt.xlabel("R")
plt.ylabel("# de veces")

plt.subplot(2,2,4)
plt.hist(caminoC)
plt.title("Histograma de C")
plt.xlabel("C")
plt.ylabel("# de veces")

plt.tight_layout()
plt.savefig("Opcionales.pdf")






