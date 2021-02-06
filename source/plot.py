import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def plot_history(array, title, scale='log', figsize=(6,3)):
    plt.figure(figsize=figsize)
    plt.title(title)
    
    for key in array:
        array[key]['total'].plot(label=key, lw=0.8)

    plt.yscale(scale)
    plt.legend(prop={'size': 8})
    plt.xlabel("")
    plt.show()
    
def plot_results(array, title, sym, scale='linear', figsize=(6,3)):
    plt.figure(figsize=figsize)
    plt.title(title)
    
    for key in array:
        array[key].query('sym == @sym')['price'].plot(label=key, lw=0.8)
        try:
            array[key].query('sym == @sym')['pred'].plot(label=key+' Prediction', lw=0.8)
        except:
            pass
    
    plt.yscale(scale)
    plt.legend(prop={'size': 8})
    plt.xlabel("")
    plt.show()