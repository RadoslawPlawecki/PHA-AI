"""
@author: Radosław Pławecki
"""


import matplotlib.pyplot as plt
import seaborn as sns

def use_latex():
    """
    Function to use LaTeX formatting for plots.
    """
    sns.set_style('whitegrid')   

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"""
            \usepackage[T1]{fontenc}
            \usepackage{lmodern}
        """
    })
