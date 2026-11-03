import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import heatmap

def plot_varianta(alpha:np.ndarray,scal=True,procent_minimal=80):
    m = len(alpha)
    x = np.arange(1,m+1)
    f = plt.figure("Scree-Plot",figsize=(9,6))
    ax = f.add_subplot(1,1,1)
    ax.set_title("Plot varianta",color="b",fontsize = 16)
    ax.set_xlabel("Componenta",fontsize=12)
    ax.set_ylabel("Varianta", fontsize=12)
    ax.set_xticks(x)
    ax.plot(x,alpha)
    ax.scatter(x,alpha,c="r",alpha=0.5)
    # Kaiser
    k1 = None
    if scal:
        k1 = len(np.where(alpha>1)[0])
        ax.axhline(1,c="g",label="Criteriul Kaiser")
    # Criteriul procentului minimal
    procent_cumulat = np.cumsum( alpha*100/sum(alpha) )
    k2 = np.where( procent_cumulat>procent_minimal )[0][0]+1
    ax.axhline(alpha[k2-1],c="m",label="Procent minimal ("+str(procent_minimal)+")")
    # Criteriul Cattell
    eps = alpha[:m-1] - alpha[1:]
    sigma = eps[:m-2] - eps[1:]
    k3 = None
    negative = sigma<0
    if any(negative):
        k3 = np.where(negative)[0][0]+2
        ax.axhline(alpha[k3 - 1], c="c", label="Criteriul Cattell (elbow)")
    ax.legend()
    plt.savefig("graphics/PlotVarianta.png")
    return k1,k2,k3


def show():
    plt.show()

def corelograma(t:pd.DataFrame,titlu="Corelograma",vmin=-1,cmap="RdYlBu",annot=True):
    f = plt.figure(titlu, figsize=(9, 6))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(titlu, color="b", fontsize=16)
    heatmap(t,vmin=vmin,vmax=1,cmap=cmap,annot=annot,ax=ax)
    plt.savefig("graphics/"+titlu+".png")

def f_scatterplot(t:pd.DataFrame,
                  x="c1",
                  y="c2",
                  titlu = "Plot scoruri",
                  corelatii=False):
    f = plt.figure(titlu, figsize=(9, 6))
    ax = f.add_subplot(1, 1, 1, aspect=1)
    ax.set_title(titlu, color="b", fontsize=16)
    if corelatii:
        pas=0.05
        theta = np.arange(0,np.pi*2+pas,pas)
        ax.plot(np.cos(theta),np.sin(theta))
        ax.plot(np.cos(theta)*0.7,np.sin(theta)*0.7,c="g")
    ax.axhline(0,c="k")
    ax.axvline(0,c="k")
    ax.scatter(t[x],t[y],c="r")
    ax.set_xlabel(x,fontdict={"fontsize":12})
    ax.set_ylabel(y,fontdict={"fontsize":12})
    n = len(t)
    for i in range(n):
        ax.text( t[x].iloc[i], t[y].iloc[i], t.index[i])
    plt.savefig("graphics/"+titlu+".png")
