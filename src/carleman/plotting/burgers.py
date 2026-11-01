
import numpy as np
import sympy as sp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation,LogLocator,NullFormatter
from carleman.plotting.style import PLOT_WIDTH,plot_style,save_figure


def sine_figure(data,output):
    x=np.unique(np.r_[np.linspace(-np.pi,np.pi,1201),0.])
    # sample closer to the symmetry and boundary zeros for the error curves
    approach=np.geomspace(1e-8,.005,64)
    error_x=np.unique(np.r_[x,-approach,approach,-np.pi+approach,np.pi-approach])
    evaluate=sp.lambdify(sp.symbols('x t',real=True),data['expression'],'numpy')
    grid=np.r_[-np.pi,data['grid'],np.pi]
    with plt.rc_context(plot_style()):
        figure,axes=plt.subplots(2,3,figsize=(PLOT_WIDTH,5.6))
        figure.subplots_adjust(left=.10,right=.91,bottom=.25,top=.94,wspace=.72,hspace=.52)
        for j,time in enumerate(data['times']):
            reference=data['reference'](x,time)
            continuous=evaluate(x,time)
            discrete=np.r_[0.,data['curves'][data['orders'][-1]][j],0.]
            top,bottom=axes[:,j]
            top.plot(x,reference,color='#0072B2',lw=1.3,label='Analytical')
            top.plot(x,continuous,color='#fd9141',lw=1.1,label='Carleman Linearised')
            top.plot(grid,discrete,color='#973003',lw=1.,ls='--',label=r'Carleman Linearised $\to$ Discretised')
            limit=1.1*max(abs(reference).max(),abs(continuous).max(),abs(discrete).max())
            top.set(ylim=(-limit,limit),title=f'$t={time:g}$')
            if j==0:top.set_ylabel('$u(x,t)$')
            right=bottom.twinx()
            exact=data['reference'](error_x,time)
            # locate the reference peak first, then refine it between nearby samples
            # all pointwise errors at this time use that same peak as denominator
            bracket=np.linspace(-np.pi,np.pi,4097)
            k=int(np.argmax(abs(data['reference'](bracket,time))))
            peak=minimize_scalar(lambda p:-abs(data['reference'](p,time)),bounds=bracket[[max(0,k-1),min(k+1,4096)]],method='bounded')
            maximum=max(abs(data['reference'](bracket,time)).max(),-peak.fun)
            errors=[]
            for values,linestyle in ((evaluate(error_x,time),'-'),(np.interp(error_x,grid,discrete),'--')):
                absolute=abs(values-exact)
                # these zeros are exact by symmetry and the boundary conditions
                # remove roundoff there before plotting on the log axis
                absolute[np.isin(error_x,[-np.pi,0.,np.pi])]=0.
                bottom.plot(error_x,np.ma.masked_less_equal(absolute,0.),color='.5',ls=linestyle,lw=1.)
                relative=100*absolute/maximum
                right.plot(error_x,relative,color='black',ls=linestyle,lw=1.)
                errors.append(relative)
            bottom.set_yscale('log',nonpositive='mask')
            bottom.yaxis.set_major_locator(LogLocator(base=10,numticks=5))
            bottom.yaxis.set_major_formatter(LogFormatterSciNotation(base=10,labelOnlyBase=True))
            bottom.yaxis.set_minor_formatter(NullFormatter())
            upper=100 if j<2 else max(100,50*np.ceil(1.05*np.max(errors)/50))
            right.set_ylim(0,upper)
            right.set_yticks(np.arange(25 if upper==100 else 50,upper+1,25 if upper==100 else 50))
            if j==0:bottom.set_ylabel('Absolute error')
            if j==2:right.set_ylabel('Relative error (%)')
            for axis in (top,bottom):
                axis.set(xlim=(-np.pi,np.pi),xlabel='$x$')
                axis.set_xticks([-np.pi,0,np.pi],[r'$-\pi$','$0$',r'$\pi$'])
        handles,labels=axes[0,0].get_legend_handles_labels()
        figure.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.075),columnspacing=1.1)
        figure.legend(handles=[Line2D([],[],color='.5',label='Absolute error'),Line2D([],[],color='black',label='Relative error')],
                      loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.015))
        for suffix in ('.pdf','.png'):save_figure(figure,output.with_suffix(suffix),dpi=300,bbox_inches='tight')
        plt.close(figure)


def rational_figure(data,output):
    grid=np.r_[-1.,data['grid'],1.]
    with plt.rc_context(plot_style()):
        figure,axes=plt.subplots(2,2,figsize=(12.2,8.))
        colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
        for j,(axis,time) in enumerate(zip(axes.flat,data['times'])):
            curves=[]
            for colour,order in zip(colours,data['orders']):
                values=np.r_[0.,data['curves'][order][j],0.]
                curves.append(values)
                axis.plot(grid,values,color=colour,lw=2.2,label=f'N = {order}')
            reference=data['reference'](grid,time)
            axis.plot(grid,reference,color=colours[4],lw=2.2,label='Exact')
            lower=min(reference.min(),*(v.min() for v in curves));upper=max(reference.max(),*(v.max() for v in curves))
            margin=.12*(upper-lower)
            axis.set(xlim=(-1,1),ylim=(lower-margin,upper+margin),title=f't = {time:g}',xlabel='x',ylabel='y(t,x)')
            axis.legend(loc='best');axis.grid(alpha=.2)
        figure.tight_layout()
        for suffix in ('.pdf','.png'):save_figure(figure,output.with_suffix(suffix),dpi=180,bbox_inches='tight')
        plt.close(figure)
