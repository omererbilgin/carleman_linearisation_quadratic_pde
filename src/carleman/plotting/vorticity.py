
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm,ListedColormap
from matplotlib.lines import Line2D
from carleman.problems.navier_stokes import velocity_from_vorticity_coefficients
from carleman.plotting.style import PLOT_WIDTH,plot_style,save_figure


def save(figure,prefix):
    for suffix in ('.pdf','.png'):save_figure(figure,prefix.with_suffix(suffix),dpi=240,bbox_inches='tight')
    plt.close(figure)


def velocity_grid(modes,coefficients,size):
    # numpy's inverse FFT divides by size**2, restore the Fourier-series normalisation
    spectrum=np.zeros((*coefficients.shape[:-2],size,size,2),dtype=complex)
    # negative wave numbers belong at the wrapped FFT indices
    spectrum[...,modes[:,0]%size,modes[:,1]%size,:]=coefficients
    values=np.fft.ifft2(spectrum,axes=(-3,-2))*size**2
    if abs(values.imag).max()>2e-8:raise ValueError('velocity reconstruction has a nonzero imaginary part')
    return values.real


def order_colourbar(figure,colours,orders,position):
    colourbar=figure.colorbar(plt.cm.ScalarMappable(norm=BoundaryNorm(np.arange(len(orders)+1)-.5,len(orders)),
                             cmap=ListedColormap(colours)),cax=figure.add_axes(position),orientation='horizontal',ticks=np.arange(len(orders)))
    colourbar.ax.set_xticklabels(list(map(str,orders)))
    colourbar.ax.tick_params(labelsize=9,length=0,pad=3)
    colourbar.set_label(r'Carleman order $N$',fontsize=9,labelpad=4)
    colourbar.ax.xaxis.set_label_position('top');colourbar.outline.set_linewidth(.5)


def velocity_figures(times,modes,trajectories,reference,prefix,quadrature=288,plot_size=100):
    orders=tuple(trajectories)
    stop=np.searchsorted(times,1.,side='right')
    times=times[:stop]
    snapshots=np.array([np.argmin(abs(times-time)) for time in (.1,.5,1.)])
    velocities=velocity_from_vorticity_coefficients(reference['modes'],reference['omega'][:stop])
    ref_grid=velocity_grid(reference['modes'],velocities,quadrature)
    ref_speed=np.linalg.norm(ref_grid,axis=-1)
    denominator=np.linalg.norm(ref_speed,axis=(-2,-1))
    ref_snap=velocity_grid(reference['modes'],velocities[snapshots],plot_size)
    # repeat the periodic edge for plotting only, keep it out of the norm grid
    speeds=[np.pad(np.linalg.norm(ref_snap,axis=-1),((0,0),(0,1),(0,1)),mode='wrap')]
    surface_orders=(3 if 3 in orders else orders[0],orders[-1])
    errors=[]
    for order,omega in trajectories.items():
        velocity=velocity_from_vorticity_coefficients(modes,omega[:stop])
        speed=np.linalg.norm(velocity_grid(modes,velocity,quadrature),axis=-1)
        # compare magnitudes here, this differs from the norm of the vector difference
        # both fields use the same uniform grid, its quadrature weight cancels
        errors.append(np.linalg.norm(speed-ref_speed,axis=(-2,-1))/denominator)
        if order in surface_orders:
            field=velocity_grid(modes,velocity[snapshots],plot_size)
            speeds.append(np.pad(np.linalg.norm(field,axis=-1),((0,0),(0,1),(0,1)),mode='wrap'))
    coordinates=np.linspace(0,2*np.pi,plot_size+1)
    x,y=np.meshgrid(coordinates,coordinates,indexing='xy')
    methods=[('Reference',speeds[0])]+[(f'Carleman $N={n}$',v) for n,v in zip(surface_orders,speeds[1:])]
    colours=plt.get_cmap('Oranges')(np.linspace(.34,.92,len(orders)))
    with plt.rc_context(plot_style({'font.size':9})):
        figure=plt.figure(figsize=(PLOT_WIDTH,5.2))
        grid=figure.add_gridspec(2,3,height_ratios=(1.,.8),left=.04,right=.96,bottom=.20,top=.96,hspace=.40,wspace=.03)
        for column,(label,values) in enumerate(methods):
            axis=figure.add_subplot(grid[0,column],projection='3d')
            limit=max(float(values[-1].max()),np.finfo(float).eps)
            # stored axes are x,y, meshgrid with xy indexing needs their transpose
            axis.plot_surface(x,y,values[-1].T,cmap='RdBu_r',vmin=0,vmax=limit,linewidth=0,rasterized=True,rcount=51,ccount=51)
            axis.set(zlim=(0,1.05*limit),xlabel='$x$',ylabel='$y$',title=label)
            axis.set_zlabel(r'$|\mathbf{v}|$' if column==0 else '',rotation=0,labelpad=1)
            for setter in (axis.set_xticks,axis.set_yticks):setter([0,np.pi,2*np.pi],['$0$',r'$\pi$',r'$2\pi$'])
            axis.view_init(elev=25,azim=-135);axis.set_box_aspect((1,1,.65));axis.tick_params(labelsize=7,pad=0);axis.grid(False)
        axis=figure.add_subplot(grid[1,:])
        for order,error,colour in zip(orders,errors,colours):axis.plot(times,100*error,color=colour,lw=1.)
        axis.set(xlim=(0,1),xlabel='$t$',ylabel='Relative error (%)',ylim=(0,max(1e-4,1.25*np.max(errors)*100)))
        axis.set_yscale('symlog',linthresh=1e-4,linscale=.7);axis.spines[['top','right']].set_visible(False);axis.grid(axis='y',color='.9',lw=.5)
        order_colourbar(figure,colours,orders,(.19,.038,.67,.026))
        save(figure,prefix)
        figure,axes=plt.subplots(3,3,figsize=(PLOT_WIDTH,7.2))
        figure.subplots_adjust(left=.13,right=.88,bottom=.07,top=.95,wspace=.12,hspace=.20)
        limit=max(values.max() for _,values in methods)
        for row,(label,values) in enumerate(methods):
            for column,time in enumerate((.1,.5,1.)):
                axis=axes[row,column]
                filled=axis.contourf(coordinates,coordinates,values[column].T,levels=np.linspace(0,limit,41),cmap='RdBu_r')
                axis.contour(coordinates,coordinates,values[column].T,levels=np.linspace(0,limit,11),colors='.25',linewidths=.3,alpha=.3)
                axis.set_aspect('equal')
                axis.set_xticks([0,np.pi,2*np.pi],['$0$',r'$\pi$',r'$2\pi$']);axis.set_yticks([0,np.pi,2*np.pi],['$0$',r'$\pi$',r'$2\pi$'])
                axis.tick_params(labelsize=8)
                if row==0:axis.set_title(f'$t={time:g}$')
                if row==2:axis.set_xlabel('$x$')
                else:axis.set_xticklabels([])
                if column==0:
                    axis.set_ylabel('$y$');axis.text(-.37,.5,label,transform=axis.transAxes,rotation=90,ha='center',va='center',fontsize=9)
                else:axis.set_yticklabels([])
        figure.colorbar(filled,cax=figure.add_axes((.91,.12,.018,.76)),label=r'$|\mathbf{v}|$')
        save(figure,prefix.with_name(prefix.name+'_contours'))
    return np.array(errors)


def palinstrophy_figure(times,histories,reference,prefix):
    orders=np.arange(2,len(histories)+2)
    colours=plt.get_cmap('Oranges')(np.linspace(.34,.92,len(orders)))
    with plt.rc_context(plot_style()):
        figure,axes=plt.subplots(2,1,figsize=(PLOT_WIDTH,6.6),sharex=True)
        figure.subplots_adjust(left=.13,right=.97,top=.96,bottom=.27,hspace=.25)
        for order,values,colour in zip(orders,histories,colours):
            axes[0].plot(times,values,color=colour,lw=1.1)
            errors=100*abs(values-reference)/reference
            axes[1].plot(times,np.ma.masked_less_equal(errors,0),color=colour,lw=1.1)
        axes[0].plot(times,reference,color='#0072B2',lw=1.3,ls=(0,(5,3)),zorder=10)
        axes[0].set_ylabel(r'$\mathcal{P}(t)$')
        axes[1].set(xlabel='$t$',ylabel='Relative error (%)',xlim=(0,1));axes[1].set_yscale('log')
        figure.legend(handles=[Line2D([],[],color='#0072B2',ls='--',label='Reference'),Line2D([],[],color=colours[min(2,len(colours)-1)],label='Carleman')],
                      loc='lower center',bbox_to_anchor=(.5,.13),ncol=2,frameon=False)
        order_colourbar(figure,colours,orders,(.19,.045,.67,.022))
        save(figure,prefix)


def tse_figures(history,cases,output):
    times=np.array(history['times']);original=np.array(history['total_palinstrophy'])
    rows=[row for row in history['rows'] if row['beta']>2 and np.any(row['palinstrophy_response'])]
    betas=[row['beta'] for row in rows]
    colours=plt.get_cmap('Oranges')(np.linspace(.34,.92,len(rows)+1))[1:]
    with plt.rc_context(plot_style()):
        figure,axis=plt.subplots(figsize=(PLOT_WIDTH,4.3))
        # this log plot shows positive responses only, the signed values stay in history
        for row,colour in zip(rows,colours):axis.plot(times,np.ma.masked_less_equal(row['palinstrophy_response'],0),color=colour,label=f"$\\beta={row['beta']}$")
        axis.set(xlabel='$t$',ylabel=r'$\mathcal{P}_{\beta}(t)$',xlim=(0,1));axis.set_yscale('log');axis.legend(frameon=False)
        save(figure,output/'palinstrophy_source_responses')
        figure,axes=plt.subplots(1,2,figsize=(PLOT_WIDTH,4.1),sharex=True,sharey=True)
        figure.subplots_adjust(left=.10,right=.98,bottom=.27,top=.91,wspace=.12)
        for axis,name in zip(axes,('increase','decrease')):
            for beta,colour in zip(betas,colours):
                case=next(c for c in cases if c['beta']==beta and c['name']==name)
                axis.plot(times,case['altered_palinstrophy'],color=colour,lw=1.)
            axis.plot(times,original,color='#0072B2',lw=1.1,zorder=3)
            axis.set(xlim=(0,1),xlabel='$t$');axis.set_xticks([0,.5,1])
        axes[0].set_ylabel(r'$\mathcal{P}(t)$')
        figure.legend(handles=[Line2D([],[],color='#0072B2',label='Original')]+[Line2D([],[],color=c,label=f'$\\beta={b}$') for b,c in zip(betas,colours)],
                      loc='lower center',ncol=len(betas)+1,frameon=False)
        save(figure,output/'palinstrophy_sequence_removal_comparison')
