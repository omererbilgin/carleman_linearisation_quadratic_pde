# vorticity plots and TSE analysis using framework in src

import argparse
import csv
from fractions import Fraction
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.sparse import save_npz
from carleman.analysis.support import supported_vorticity_matrix
from carleman.analysis.flows import reference_history,supported_history,palinstrophy_history,kinetic_energy_table
from carleman.analysis import tse
from carleman.analysis.eigenvalue_removal import compute as eigenvalue_removal
from carleman.problems.fourier_vorticity import fourier_vorticity_operators
from carleman.plotting.vorticity import velocity_figures,palinstrophy_figure,tse_figures


def table(headers,rows,path):
    with path.open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(headers);writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description='Vorticity figures, operators and the three article tables')
    parser.add_argument('path',choices=('all','flow','tse'),nargs='?',default='all')
    parser.add_argument('--output',type=Path,default=Path('results'))
    parser.add_argument('--quick',action='store_true',help='R=3,N=4 and small reference grids, not the article truncation')
    args=parser.parse_args()
    radius,order,grid,samples=(3,4,12,21) if args.quick else (30,10,144,201)
    figures,data=args.output/'figures',args.output/'data'
    figures.mkdir(parents=True,exist_ok=True);data.mkdir(parents=True,exist_ok=True)
    operators=fourier_vorticity_operators(radius,.001)
    for name,matrix in [('F1',operators.F1),('F2',operators.F2)]:save_npz(data/f'vorticity_R{radius}_{name}.npz',matrix)
    del operators
    problem=supported_vorticity_matrix(order,radius,Fraction(2),.001)
    save_npz(data/f'vorticity_R{radius}_N{order}_C.npz',problem['matrix'])
    if args.path!='tse':
        times=np.linspace(0.,2.,samples)
        reference=reference_history(times,grid,2.,data)
        trajectories={}
        for n in range(2,order+1):
            cache=data/f'vorticity_R{radius}_N{n}_T2_M{samples}.npz'
            if not cache.exists():
                np.savez_compressed(cache,omega=supported_history(problem,times,n))
            with np.load(cache) as saved:trajectories[n]=saved['omega'].copy()
        stem=f'vorticity_velocity_R{radius}_N2_{order}_eps2_nu0p001_T2_M{samples}_G{grid}_view_T1_magnitude'
        errors=velocity_figures(times,problem['modes'],trajectories,reference,figures/stem,quadrature=48 if args.quick else 288,plot_size=32 if args.quick else 100)
        np.savetxt(data/'velocity_errors.csv',np.column_stack((times[:errors.shape[1]],100*errors.T)),delimiter=',',header='time,'+','.join(f'N{n}_percent' for n in trajectories),comments='')
        ptimes=np.linspace(0.,1.,10 if args.quick else 50)
        histories=palinstrophy_history(order,radius,ptimes)
        ref=reference_history(ptimes,grid if args.quick else 72,2.,data)
        reference_P=2*np.pi**2*(abs(ref['omega'])**2@np.sum(ref['modes']**2,axis=1))
        palinstrophy_figure(ptimes,histories,reference_P,figures/f'vorticity_path_palinstrophy_a_R{radius}_T1_M{len(ptimes)}_eps2_nu0p001')
        np.savetxt(data/'palinstrophy.csv',np.column_stack((ptimes,reference_P,histories.T)),delimiter=',',header='time,reference,'+','.join(f'N{n}' for n in trajectories),comments='')
        table(['epsilon','t0_percent','t0.1_percent','t0.5_percent','t1_percent'],kinetic_energy_table(data,grid),data/'kinetic_energy_table.csv')
    if args.path!='flow':
        cache=data/f'TSE_R{radius}_N{order}_eps2_nu0p001.json'
        sector=tse.supported_problem(order,radius,Fraction(2))
        if not cache.exists():
            counts,repetitions,visited=tse.count_sequences(sector)
            residues=tse.cumulative_residues(tse.observable_resolvent_by_degree(sector),order)
            report=tse.make_report(sector,counts,repetitions,visited,residues,Fraction(1,1000))
            cache.write_text(json.dumps(report,indent=2)+'\n')
        report=json.loads(cache.read_text())
        counts={int(n):int(value) for n,value in report['counts_by_source_degree'].items()}
        betas=list(range(2,order+1,2))
        table(['N',*(f'beta{b}' for b in betas),'total'],[[f'{n}--{n+1}' if n<order else str(n),*(counts.get(b,0) if b<=n else '--' for b in betas),sum(counts.get(b,0) for b in betas if b<=n)] for n in range(2,order+1,2)],data/'TSE_counts.csv')
        history=tse.source_history(report,np.linspace(0.,1.,201))
        cases=tse.selection_histories(sector,counts,history,percent=1.)
        tse_figures(history,cases,figures)
        (data/'TSE_selection.json').write_text(json.dumps(cases,indent=2)+'\n')
        eigen_cache=data/f'eigenvalue_removal_R{radius}_N{order}.json'
        if not eigen_cache.exists():
            eigen_cache.write_text(json.dumps(eigenvalue_removal(order=order,radius=radius,epsilon=Fraction(2),problem=problem),indent=2)+'\n')
        ranking=json.loads(eigen_cache.read_text())['groups']
        table(['rank','lambda_omega/nu','omega_error_percent','lambda_P/nu','P_error_percent'],
              [[i+1,-w['unit_viscosity_decay_rate'],w['relative_removal_error_percent'],-p['unit_viscosity_decay_rate'],p['relative_removal_error_percent']]
               for i,(w,p) in enumerate(zip(ranking[0]['ranking'][:10],ranking[1]['ranking'][:10]))],data/'eigenvalue_table.csv')


if __name__=='__main__':main()
