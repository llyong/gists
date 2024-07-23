
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

import matplotlib

matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage[utf8]{inputenc}"]


framableu = (12,91,122)
framableulight = (18,144,176)
framavert = (142,156,72)
framavertlight = (227,235,199)
framarouge = (204,45,24)
framarougelight = (249,189,187)
framaviolet = (106,86,135)
framavioletlight = (211,197,232)
framaorange = (235,114,57)
framaorangelight = (235,209,197)
framajaune = (196,168,27)
framajaunelight = (255,235,181)
framamarron = (161,136,127)
framamarronlight = (215,204,200)
framagris = (97,97,97)
framagrislight = (245,245,245)

framableu = tuple(c/255 for c in framableu)
framableulight = tuple(c/255 for c in framableulight)
framavert = tuple(c/255 for c in framavert)
framavertlight = tuple(c/255 for c in framavertlight)
framarouge = tuple(c/255 for c in framarouge)
framarougelight = tuple(c/255 for c in framarougelight)
framaviolet = tuple(c/255 for c in framaviolet)
framavioletlight = tuple(c/255 for c in framavioletlight)
framaorange = tuple(c/255 for c in framaorange)
framaorangelight = tuple(c/255 for c in framaorangelight)
framajaune = tuple(c/255 for c in framajaune)
framajaunelight = tuple(c/255 for c in framajaunelight)
framamarron = tuple(c/255 for c in framamarron)
framamarronlight = tuple(c/255 for c in framamarronlight)
framagris = tuple(c/255 for c in framagris)
framagrislight = tuple(c/255 for c in framagrislight)

colors = [framableu, framableulight, framavert, framavertlight,
          framarouge, framarougelight, framaviolet, framavioletlight,
          framaorange, framaorangelight, framajaune, framajaunelight,
          framamarron, framamarronlight, framagris, framagrislight]



def step_fct(z):
    return( 0. if z<0 else 1. )

def linear_fct(z):
    return( z )

def sigmoid_fct(z):
    return( 1/(1+np.exp(-1*z)) )

def tanh_fct(z):
    return( np.tanh(z) )

def relu_fct(z):
    return( 0. if z<0 else z )

def leaky_relu_fct(z):
    return( 0.1*z if z<0 else z )

def parametrised_relu_fct(z):
    return( 0.05*z if z<0 else 0.5*z )

def elu_fct(z):
    return( (np.exp(z)-1) if z<0 else z )

def swish_fct(z):
    return( z*sigmoid_fct(z) )

def sinerelu_fct(z):
    return( 0.1*(np.sin(z)-np.cos(z)) if z<0 else z)

activation_fcts = [step_fct, linear_fct, sigmoid_fct, tanh_fct, relu_fct,
                   leaky_relu_fct, parametrised_relu_fct, elu_fct, swish_fct, sinerelu_fct]




def plot_activation_fcts(activation_fcts):
    x_min = -10.
    x_max = 10.
    y_min = -2.
    y_max = 5.
    lim_space = 0.04
    x_num = 100

    zs = np.linspace(x_min, x_max, x_num)

    fig, ax = plt.subplots(figsize=(10, 6))
    for (i, activation_fct) in enumerate(activation_fcts):
        ax.plot(zs, [activation_fct(z) for z in zs], color=colors[i],
                label=activation_fct.__name__)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # Set limits of axes
    #ax.set_xlim(x_min-(x_max-x_min)*lim_space, x_max+(x_max-x_min)*lim_space)
    ax.set_ylim(y_min-(y_max-y_min)*lim_space, y_max+(y_max-y_min)*lim_space)

    # Set labels
    ax.set_xlabel(r'$z$', fontsize='x-large')
    ax.set_ylabel(r'$\sigma(z)$', fontsize='x-large')

    # Show grid
    ax.grid(True)

    # Show legend
    ax.legend(fontsize='large', loc=2)#4)

    #plt.show()
    #return( ax )


def plot_activation_fcts_(activation_fcts):
    x_min = -10.
    x_max = 10.
    y_min = -2.
    y_max = 5.
    lim_space = 0.04
    x_num = 100

    zs = np.linspace(x_min, x_max, x_num)

    fig, axs = plt.subplots(2, 5, figsize=(10, 6))
    for (i, activation_fct) in enumerate(activation_fcts):
        ax = axs[i%2, i//2]
        ax.plot(zs, [activation_fct(z) for z in zs], color=colors[i])#, label=activation_fct.__name__)

        ax.set_title(activation_fct.__name__)

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        # Set limits of axes
        ax.set_xlim(x_min-(x_max-x_min)*lim_space, x_max+(x_max-x_min)*lim_space)
        ax.set_ylim(y_min-(y_max-y_min)*lim_space, y_max+(y_max-y_min)*lim_space)

        # Set labels
        if i%2 == 1:
            ax.set_xlabel(r'$z$', fontsize='large')
        else:
            ax.set_xticklabels([])
        if i//2 == 0:
            ax.set_ylabel(r'$\sigma(z)$', fontsize='large')
        else:
            ax.set_yticklabels([])

        # Show grid
        ax.grid(True)

    #plt.savefig('Plots/activation_fcts.png')
    #plt.show()
    #return( ax )





def shift_scale_fct(z, shift=0., scale=1.):
    return( scale*z+shift )


def plot_sigmoid(ax, shift=0., scale=1.):
    x_min = -10. # 0.
    x_max = 10.
    y_min = 0.
    y_max = 1.
    lim_space = 0.04
    x_num = 100

    zs = np.linspace(x_min, x_max, x_num)
    sigmoids = [sigmoid_fct(shift_scale_fct(z, shift, scale)) for z in zs]

    # fig, ax = plt.subplots()
    ax.plot(zs, sigmoids, color=framableu,
            label=r'$w='+str(int(scale))+r'$'+'\n'+r'$b='+str(int(shift))+r'$')
    #ax.plot(zs, sigmoids, color=framableu,
    #        label=r'$\sigma_{w,b}(x)=\frac{1}{1+e^{-('+str(int(scale))+r'x+'+str(int(shift))+r')}}$')

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # Set limits of axes
    ax.set_xlim(x_min-(x_max-x_min)*lim_space, x_max+(x_max-x_min)*lim_space)
    ax.set_ylim(y_min-(y_max-y_min)*lim_space, y_max+(y_max-y_min)*lim_space)

    # Set labels
    ax.set_xlabel(r'$x$', fontsize='x-large')
    ax.set_ylabel(r'$\sigma_{wb}(x)$', fontsize='x-large')

    # Show grid
    ax.grid(True)

    # Show legend
    ax.legend(fontsize='large', loc=2, handlelength=0.)

    ##plt.show()
    #return( ax )












def sigmoid_fct_wb(x, w, b):
    return( 1/(1+np.exp(-(w*x+b))) )

def sigmoid_fct_1d(x, ws, bs):
    return( np.sum([sigmoid_fct_wb(x, w, b) for (w, b) in zip(ws, bs)]) )

def sigmoid_fct_2d_wb(x, y, ws, bs):
    return( sigmoid_fct_1d(x, ws[0], bs[0]) + sigmoid_fct_1d(y, ws[1], bs[1]) )

def plot_sigmoid_fct_1d_wb(ax, ws, bs):
    x_min = 0.
    x_max = 1.
    y_min = 0.
    y_max = 2.
    lim_space = 0.04
    x_num = 100

    xs = np.linspace(x_min, x_max, x_num)

    len_s = 5
    ss = [-b/w for (w, b) in zip(ws, bs)]
    label = ''
    for (i, (w, b, s)) in enumerate(zip(ws, bs, ss)):
        label += r'$w_'+str(i)+r'='+str(int(w))+r'$'+'\n'+\
                    r'$b_'+str(i)+r'='+str(int(b))+r'$'+'\n'+\
                    r'$s_'+str(i)+r'='+str(s)[:len_s]+r'$'
        if i != len(ws)-1:
            label += '\n'
    ax.plot(xs, [sigmoid_fct_1d(x, ws, bs) for x in xs], color=framableu,
            label=label)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # Set limits of axes
    #ax.set_xlim(x_min-(x_max-x_min)*lim_space, x_max+(x_max-x_min)*lim_space)
    ax.set_ylim(y_min-(y_max-y_min)*lim_space, y_max+(y_max-y_min)*lim_space)

    # Set labels
    ax.set_xlabel(r'$x$', fontsize='large')
    ax.set_ylabel(r'$\sum_i\sigma_{wb,i}(x)$', fontsize='large')

    # Show grid
    ax.grid(True)

    # Show legend
    ax.legend(loc=2, handlelength=0.)#, fontsize='large')


def plot_sigmoid_fct_2d_wb(ws, bs):
    x_min = 0.
    x_max = 1.
    y_min = 0.
    y_max = 2.
    lim_space = 0.04
    x_num = 50

    xs = np.linspace(x_min, x_max, x_num)

    Xs, Ys = np.meshgrid(xs, xs)

    fig = plt.figure()
    ax = plt.gca(projection='3d')

    ax.plot_surface(Xs, Ys, np.array([[sigmoid_fct_2d(x, y, ws, bs) for (x, y) in zip(X, Y)] for (X, Y) in zip(Xs, Ys)]))

    return( fig )





def sigmoid_fct_s_1d(x, ss):
    w = 600.
    return( np.sum([sigmoid_fct(x, w, -s*w) for s in ss]) )

def plot_sigmoid_fct_s_1d(ax, ss):
    x_min = 0.
    x_max = 1.
    y_min = 0.
    y_max = 2.
    lim_space = 0.04
    x_num = 100

    xs = np.linspace(x_min, x_max, x_num)
    w=600.

    len_s = 5
    label = ''
    for (i, (w, b, s)) in enumerate(zip(ws, bs, ss)):
        label += r'$s_'+str(i)+r'='+str(s)[:len_s]+r'$'
        if i != len(ws)-1:
            label += '\n'

    ax.plot(xs, [sigmoid_fct_s_1d(x, ss) for x in xs], color=framableu,
            label=label)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # Set limits of axes
    #ax.set_xlim(x_min-(x_max-x_min)*lim_space, x_max+(x_max-x_min)*lim_space)
    ax.set_ylim(y_min-(y_max-y_min)*lim_space, y_max+(y_max-y_min)*lim_space)

    # Set labels
    ax.set_xlabel(r'$x$', fontsize='large')
    if y_label:
        ax.set_ylabel(r'$\sigma_{wb}(x)$', fontsize='large')
    else:
        ax.set_yticklabels([])

    # Show grid
    ax.grid(True)

    # Show legend
    ax.legend(loc=2, handlelength=0.)#, fontsize='large')

def sigmoid_fct_s_wout_1d(x, ss, wouts):
    w = 600.
    return( np.sum([wout*sigmoid_fct_wb(x, w, -s*w) for (s, wout) in zip(ss, wouts)]) )

def plot_sigmoid_fct_s_wout_1d(ax, ss, ws):
    x_min = 0.
    x_max = 1.
    y_min = -2.
    y_max = 2.
    lim_space = 0.04
    x_num = 100

    xs = np.linspace(x_min, x_max, x_num)

    len_s = 5
    len_w = 5
    label = ''
    for (i, (s, w)) in enumerate(zip(ss, ws)):
        label += r'$s_'+str(i)+r'='+str(s)[:len_s]+r'$'+'\n'+\
                    r'$w_{\mathrm{out},'+str(i)+r'}='+str(w)[:len_w]+r'$'
        if i != len(ws)-1:
            label += '\n'

    ax.plot(xs, [sigmoid_fct_s_wout_1d(x, ss, ws) for x in xs], color=framableu,
            label=label)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # Set limits of axes
    #ax.set_xlim(x_min-(x_max-x_min)*lim_space, x_max+(x_max-x_min)*lim_space)
    ax.set_ylim(y_min-(y_max-y_min)*lim_space, y_max+(y_max-y_min)*lim_space)

    # Set labels
    ax.set_xlabel(r'$x$', fontsize='large')
    ax.set_ylabel(r'$A_{w, \mathrm{out}}(x)$', fontsize='large')

    # Show grid
    ax.grid(True)

    # Show legend
    ax.legend(loc=2, handlelength=0.)#, fontsize='large')










def plot_sigmoid_fct_s_h_1d(ax, ss, hs):
    x_min = 0.
    x_max = 1.
    y_min = -2.
    y_max = 2.
    lim_space = 0.04
    x_num = 100

    xs = np.linspace(x_min, x_max, x_num)

    ws = [[h, -h] for h in hs]
    ws = [j for i in ws for j in i]

    len_s = 5
    len_h = 5
    label = ''
    for (i, (s, w)) in enumerate(zip(ss, ws)):
        label += r'$s_'+str(i)+r'='+str(s)[:len_s]+r'$'
        if i%2 == 1:
            label += '\n'+r'$h_{'+str(i-1)+str(i)+r'}='+str(-w)[:len_h]+r'$'
        if i != len(ws)-1:
            label += '\n'

    ax.plot(xs, [sigmoid_fct_s_wout_1d(x, ss, ws) for x in xs], color=framableu,
            label=label)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # Set limits of axes
    #ax.set_xlim(x_min-(x_max-x_min)*lim_space, x_max+(x_max-x_min)*lim_space)
    ax.set_ylim(y_min-(y_max-y_min)*lim_space, y_max+(y_max-y_min)*lim_space)

    # Set labels
    ax.set_xlabel(r'$x$', fontsize='large')
    ax.set_ylabel(r'$A_{w, \mathrm{out}}(x)$', fontsize='large')

    # Show grid
    ax.grid(True)

    # Show legend
    ax.legend(loc=2, handlelength=0.)#, fontsize='large')





def sigmoid_fct_tower_1d(x, ss, ws):
    return( ws[0]*sigmoid_fct(x, w, -ss[0]*w) + ws[1]*sigmoid_fct(x, w, -ss[1]*w) )

def sigmoid_tower_1d(x, sss, wss):
    return( np.sum([sigmoid_fct_tower_1d(x, ss, ws) for (ss, ws) in zip(sss, wss)]) )

def plot_sigmoid_tower_1d(xs, sss, wss):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xs, [sigmoid_tower_1d(x, sss, wss) for x in xs])






def sigmoid_fct_tower_h_1d(x, ss, h):
    w = 600.
    return( h*sigmoid_fct_wb(x, w, -ss[0]*w) - h*sigmoid_fct_wb(x, w, -ss[1]*w) )

def sigmoid_tower_h_1d(x, sss, hs):
    return( np.sum([sigmoid_fct_tower_h_1d(x, ss, h) for (ss, h) in zip(sss, hs)]) )

def plot_sigmoid_tower_h_1d(ax, y_label, sss, hs):
    x_min = 0.
    x_max = 1.
    y_min = 0.
    y_max = 2.
    lim_space = 0.04
    x_num = 100

    xs = np.linspace(x_min, x_max, x_num)

    len_s = 3
    len_h = 4
    label = ''
    for (i, (s, h)) in enumerate(zip(sss, hs)):
        label += r'$s_'+str(2*i)+r'='+str(s[0])[:len_s]+r'$'+'\n'+\
                    r'$s_'+str(2*i+1)+r'='+str(s[1])[:len_s]+r'$'+'\n'+\
                    r'$h_{'+str(2*i)+str(2*i+1)+r'}='+str(h)[:len_h]+r'$'
        if i != len(hs)-1:
            label += '\n'

    ax.plot(xs, [sigmoid_tower_h_1d(x, sss, hs) for x in xs], color=framableu,
            label=label)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # Set limits of axes
    #ax.set_xlim(x_min-(x_max-x_min)*lim_space, x_max+(x_max-x_min)*lim_space)
    ax.set_ylim(y_min-(y_max-y_min)*lim_space, y_max+(y_max-y_min)*lim_space)

    # Set labels
    ax.set_xlabel(r'$x$', fontsize='large')
    if y_label:
        ax.set_ylabel(r'$A_{w, \mathrm{out}}(x)$', fontsize='large')
    else:
        ax.set_yticklabels([])

    # Show grid
    ax.grid(True)

    # Show legend
    ax.legend(loc=2, handlelength=0.)










def sigmoid_tower_num_1d(x, hs):
    num_tower = len(hs)
    width_tower = 1./num_tower
    sss = [[i, i+width_tower] for i in np.linspace(0., 1.-width_tower, num_tower)]
    return( sigmoid_tower_h_1d(x, sss, hs) )

def plot_sigmoid_tower_num_1d(xs, hs):
    num_tower = len(hs)
    width_tower = 1./num_tower
    sss = [[i, i+width_tower] for i in np.linspace(0., 1.-width_tower, num_tower)]
    plot_sigmoid_tower_h_1d(xs, sss, hs)

def call_sigmoid_tower_num_1d(x,
                             h0, h1, h2, h3, h4, h5, h6, h7, h8, h9):
    hs = [h0, h1, h2, h3, h4, h5, h6, h7, h8, h9]
    return( sigmoid_tower_num_1d(x, hs) )













'''
def sigmoid_fct_s_wout_1d(x, ss, wouts):
    return( np.sum([wout*sigmoid_fct_wb(x, w, -s*w) for (s, wout) in zip(ss, wouts)]) )
'''
def sigmoid_fct_s_wout_2d(x, y, ss, wouts):
    return( sigmoid_fct_s_wout_1d(x, ss[0], wouts[0]) + sigmoid_fct_s_wout_1d(y, ss[1], wouts[1]) )


def plot_sigmoid_fct_s_wout_2d(sss, woutss):
    x_min = 0.
    x_max = 1.
    y_min = 0.
    y_max = 2.
    lim_space = 0.04
    x_num = 50

    xs = np.linspace(x_min, x_max, x_num)


    Xs, Ys = np.meshgrid(xs, xs)

    fig = plt.figure()
    ax = plt.gca(projection='3d')

    len_s = 4
    len_wout = 5
    ds = ['x', 'y']

    label = ''

    for (j, (d, ss, wouts)) in enumerate(zip(ds, sss, woutss)):
        for (i, (s, wout)) in enumerate(zip(ss, wouts)):
            label += r'$s_{'+d+r','+str(i)+r'}='+str(s)[:len_s]+r'$'+'\n'+\
                            r'$w_{\mathrm{out},'+d+r','+str(i)+r'}='+str(wout)[:len_wout]+r'$'

            if i < len(wouts)-1:
                label += '\n'

    label = '$\n$'.join(label.split('$$'))


    Zs = np.array([[sigmoid_fct_s_wout_2d(x, y, sss, woutss) for (x, y) in zip(X, Y)] for (X, Y) in zip(Xs, Ys)])

    p = ax.plot_surface(Xs, Ys, Zs, label=label, rcount=100, ccount=100)

    # Fixes bug that occurs by using legend
    p._facecolors2d = p._facecolors3d
    p._edgecolors2d = p._edgecolors3d

    # Set distance of viewer
    ax.dist = 13

    # Axes labels
    ax.set_xlabel(r'$x$', fontsize='large')
    ax.set_ylabel(r'$y$', fontsize='large')
    ax.set_zlabel(r'$A_{w, \mathrm{out}}(\boldsymbol{x})$', fontsize='large')

    # Add legend
    ax.legend(loc=2, handlelength=0., fontsize='large')

    plt.tight_layout()

    #plt.savefig('Plots/plot_sigmoid_fct_s_wout_2d_'+str(key)+'.png', dpi=200)

    #plt.show()

    return( fig )












def sigmoid_fct_h_2d(x, y, sss, hs):
    return( sigmoid_tower_h_1d(x, sss[0], hs[0]) + sigmoid_tower_h_1d(y, sss[1], hs[1]) )

def plot_sigmoid_fct_h_2d(sss, hss):
    x_min = 0.
    x_max = 1.
    y_min = 0.
    y_max = 2.
    lim_space = 0.04
    x_num = 50

    xs = np.linspace(x_min, x_max, x_num)

    Xs, Ys = np.meshgrid(xs, xs)

    fig = plt.figure()
    ax = plt.gca(projection='3d')

    len_s = 4
    len_h = 4
    ds = ['x', 'y']

    label = ''

    for (j, (d, ss, hs)) in enumerate(zip(ds, sss, hss)):
        for (i, (s, h)) in enumerate(zip(ss, hs)):
            for k in range(len(s)):
                label += r'$s_{'+d+r','+str(2*i+k)+r'}='+str(s[k])[:len_s]+r'$'+'\n'
            label += r'$h_{'+d+r','+str(2*i)+str(2*i+1)+r'}='+str(h)[:len_h]+r'$'

            if i < len(hs)-1:
                label += '\n'

    label = '$\n$'.join(label.split('$$'))

    Zs = np.array([[sigmoid_fct_h_2d(x, y, sss, hss) for (x, y) in zip(X, Y)] for (X, Y) in zip(Xs, Ys)])

    p = ax.plot_surface(Xs, Ys, Zs, label=label, rcount=100, ccount=100)

    # Fixes bug that occurs by using legend
    p._facecolors2d = p._facecolors3d
    p._edgecolors2d = p._edgecolors3d

    # Set distance of viewer
    ax.dist = 13

    # Axes labels
    ax.set_xlabel(r'$x$', fontsize='large')
    ax.set_ylabel(r'$y$', fontsize='large')
    ax.set_zlabel(r'$A_{w, \mathrm{out}}(\boldsymbol{x})$', fontsize='large')

    # Add legend
    ax.legend(loc=2, handlelength=0., fontsize='large')

    plt.tight_layout()

    hss_str = '_'.join(map(str, [i for j in hss for i in j]))

    #plt.savefig('Plots/plot_sigmoid_fct_h_2d'+hss_str+'.png', dpi=200)

    #plt.show()

    return( fig )










def sigmoid_fct_h_wout_b_2d(x, y, sss, hs, b):
    X = 1.
    w = sigmoid_tower_h_1d(x, sss[0], hs[0]) + sigmoid_tower_h_1d(y, sss[1], hs[1])
    return( sigmoid_fct_wb(X, w, b) )

def plot_sigmoid_fct_h_wout_b_2d(sss, hss, b):
    x_min = 0.
    x_max = 1.
    y_min = 0.
    y_max = 2.
    lim_space = 0.04
    x_num = 50

    xs = np.linspace(x_min, x_max, x_num)

    Xs, Ys = np.meshgrid(xs, xs)

    fig = plt.figure()
    ax = plt.gca(projection='3d')

    len_s = 3
    len_h = 4
    ds = ['x', 'y']

    label = ''

    for (j, (d, ss, hs)) in enumerate(zip(ds, sss, hss)):
        for (i, (s, h)) in enumerate(zip(ss, hs)):
            for k in range(len(s)):
                label += r'$s_{'+d+r','+str(2*i+k)+r'}='+str(s[k])[:len_s]+r'$'+'\n'
            label += r'$h_{'+d+r','+str(2*i)+str(2*i+1)+r'}='+str(h)[:len_h]+r'$'

            if i < len(hs)-1:
                label += '\n'

    label = '$\n$'.join(label.split('$$'))
    label += '\n'+r'$b_{\mathrm{out}}='+str(b)+'$'


    Zs = np.array([[sigmoid_fct_h_wout_b_2d(x, y, sss, hss, b) for (x, y) in zip(X, Y)] for (X, Y) in zip(Xs, Ys)])

    p = ax.plot_surface(Xs, Ys, Zs, label=label, rcount=200, ccount=200)

    # Fixes bug that occurs by using legend
    p._facecolors2d = p._facecolors3d
    p._edgecolors2d = p._edgecolors3d

    # Set distance of viewer
    ax.dist = 13

    # Axes labels
    ax.set_xlabel(r'$x$', fontsize='large')
    ax.set_ylabel(r'$y$', fontsize='large')
    ax.set_zlabel(r'$\sigma(A_{wb, \mathrm{out}}(\boldsymbol{x}))$', fontsize='large')

    # Add legend
    ax.legend(loc=2, handlelength=0., fontsize='large')

    plt.tight_layout()

    #plt.savefig('Plots/one_tower_minus_bias_'+str(b)+'.png', dpi=200)

    #plt.show()

    return( fig )











def sigmoid_tower_2d(x, y, sss):
    len_s = len(sss)
    hs = [[10.]]*len_s
    w = -15.
    return( sigmoid_fct_h_wout_b_2d(x, y, sss, hs, w) )

def sigmoid_tower_2d_sum(x, y, ssss, ws):
    return( np.sum([w * sigmoid_tower_2d(x, y, sss) for (sss, w) in zip(ssss, ws)]) )
    #return( np.sum([sigmoid_tower_2d(x, y, sss) for sss in ssss]) )

def plot_sigmoid_fct_towers_bout_2d(ssss, wouts):
    x_min = 0.
    x_max = 1.
    y_min = 0.
    y_max = 2.
    lim_space = 0.04
    x_num = 50

    xs = np.linspace(x_min, x_max, x_num)


    Xs, Ys = np.meshgrid(xs, xs)

    fig = plt.figure()
    ax = plt.gca(projection='3d')

    len_s = 3
    len_w = 4
    ds = ['x', 'y']

    label = ''

    for (l, (d, sss, w)) in enumerate(zip(ds, ssss, wouts)):
        for (j, ss) in enumerate(sss):
            for (i, s) in enumerate(ss):
                for k in range(len(s)):
                    label += r'$s_{'+d+r','+str(2*j+k)+r'}='+str(s[k])[:len_s]+r'$'#+'\n'
                #label += r'$b_{\mathrm{out},'+d+r','+str(2*j)+str(2*j+1)+r'}='+str(b)[:len_b]+r'$'
                    #if k < len(s):
                    #    label += '\n'
                if i < len(sss)-1:
                    pass #label += '\n'

    label = '$\n$'.join(label.split('$$'))

    for (i, w) in enumerate(wouts):
        label += '\n'+r'$w_{\mathrm{out},'+str(2*i)+str(2*i+1)+r'}^\prime='+str(w)[:len_w]+r'$'


    Zs = np.array([[sigmoid_tower_2d_sum(x, y, ssss, wouts) for (x, y) in zip(X, Y)] for (X, Y) in zip(Xs, Ys)])

    p = ax.plot_surface(Xs, Ys, Zs, label=label, rcount=200, ccount=200)

    # Fixes bug that occurs by using legend
    p._facecolors2d = p._facecolors3d
    p._edgecolors2d = p._edgecolors3d

    # Set distance of viewer
    ax.dist = 13

    # Axes labels
    ax.set_xlabel(r'$x$', fontsize='large')
    ax.set_ylabel(r'$y$', fontsize='large')
    ax.set_zlabel(r'$\sum_iw_{\mathrm{out}, i}^\prime\sigma(A_{wb, \mathrm{out}, i}(\boldsymbol{x}))$', fontsize='large')

    # Add legend
    ax.legend(loc=2, handlelength=0., fontsize='large')

    plt.tight_layout()

    wouts_str = '_'.join(map(str, wouts))

    #plt.savefig('Plots/two_weighted_towers_'+wouts_str+'.png', dpi=200)

    #plt.show()

    return( fig )









def optional_activation_fct(z):
    return( 1/(1+np.exp(-z)) + 0.2 * np.exp(-z**2/4) * np.sin(5*z))

def plot_optional_activation(ax, shift=0., scale=1.):
    x_min = -5.
    x_max = 5.
    y_min = 0.
    y_max = 1.
    lim_space = 0.04
    x_num = 100

    zs = np.linspace(x_min, x_max, x_num)
    sigmoids = [optional_activation_fct(shift_scale_fct(z, shift, scale)) for z in zs]

    # fig, ax = plt.subplots()
    ax.plot(zs, sigmoids, color=framableu,
            label=r'$w='+str(int(scale))+r'$'+'\n'+r'$b='+str(int(shift))+r'$')

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # Set limits of axes
    #ax.set_xlim(x_min-(x_max-x_min)*lim_space, x_max+(x_max-x_min)*lim_space)
    #ax.set_ylim(y_min-(y_max-y_min)*lim_space, y_max+(y_max-y_min)*lim_space)

    # Set labels
    ax.set_xlabel(r'$x$', fontsize='x-large')
    ax.set_ylabel(r'$\tilde{\sigma}_{wb}(x)$', fontsize='x-large')

    # Show grid
    ax.grid(True)

    # Show legend
    ax.legend(fontsize='large', loc=2, handlelength=0.)

    ##plt.show()
    #return( ax )
