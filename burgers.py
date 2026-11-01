# burgers plots using framework in src

import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.sparse import csr_matrix,save_npz
from carleman.problems.burgers import comparison_data
from carleman.plotting.burgers import sine_figure,rational_figure


def main():
    parser=argparse.ArgumentParser(description='The two Burgers article figures and their operators')
    parser.add_argument('--output',type=Path,default=Path('results'))
    parser.add_argument('--quick',action='store_true',help='small spatial grids for a smoke run')
    args=parser.parse_args()
    figures,data=args.output/'figures',args.output/'data'
    figures.mkdir(parents=True,exist_ok=True);data.mkdir(parents=True,exist_ok=True)
    for profile,points,order,name,plot in (
        ('sine',8 if args.quick else 64,3,'burgers_sinx_carleman_comparison',sine_figure),
        ('rational',8 if args.quick else 30,4,'burgers_heinzelreiter_central_difference',rational_figure)):
        result=comparison_data(profile,points,order)
        for label,matrix in [('F1',csr_matrix(result['F1'])),('F2',csr_matrix(result['F2'].reshape(points,-1))),('C',result['matrix'])]:
            save_npz(data/f'burgers_{profile}_{label}.npz',matrix)
        plot(result,figures/name)
        np.savez_compressed(data/f'burgers_{profile}.npz',grid=result['grid'],times=result['times'],
                            **{f'N{n}':v for n,v in result['curves'].items()})


if __name__=='__main__':main()
