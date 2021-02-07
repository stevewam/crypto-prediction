"""Plotting Script

This script is used to plot results from modelling activities.

This file can also be imported as a module and contains the following
functions:

    * plot_history - plot returns from trade history
    * plot_result - plot prediction results against actual for a particular currency

"""

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

def plot_history(array, title, scale='log', figsize=(6,3)):
    """Plot returns from trade history

    Parameters
    ----------
    array : dict
        Dictionary with set name as keys and the dataset as the value
    title : str
        Plot title
    scale : str, optional
        Scale of Y-axis
    figsize: tuple, optional
        Size of figure

    """
    plt.figure(figsize=figsize)
    plt.title(title)
    
    #Plot for each item in dictionary
    for key in array:
        array[key]['total'].plot(label=key, lw=0.8)

    plt.yscale(scale)
    plt.legend(prop={'size': 8})
    plt.xlabel("")
    
    #Save plot
    plt.savefig(os.path.join('./plots', title + '.png'), dpi=200, bbox_inches='tight')
    
    plt.show()
    
    
def plot_results(array, title, sym, scale='linear', figsize=(6,3)):
    """Plot prediction results against actual for a particular currency

    Parameters
    ----------
    array : dict
        Dictionary with set name as keys and the dataset as the value
    title : str
        Plot title
    sym: str
        Symbol for currency to be plotted
    scale : str, optional
        Scale of Y-axis
    figsize: tuple, optional
        Size of figure

    """
    plt.figure(figsize=figsize)
    plt.title(title)
    
    #Plot for each item in dictionary
    for key in array:
        array[key].query('sym == @sym')['price'].plot(label=key, lw=0.8)
        
        #Plot prediction value if column exists
        try:
            array[key].query('sym == @sym')['pred'].plot(label=key+' Prediction', lw=0.8)
        except:
            pass
    
    plt.yscale(scale)
    plt.legend(prop={'size': 8})
    
    #Remove X-axis label
    plt.xlabel("")
    
    #Save plot
    plt.savefig(os.path.join('./plots', title + '.png'), dpi=200, bbox_inches='tight')
    
    plt.show()
    