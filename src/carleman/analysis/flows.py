# Algorthmic structure:
#                           The supported matrix is restricted by degree for each N, extra zero coordinates does not affect the solution
#                           Vorticity is read from degree one, palinstrophy is read from degree two in its zero-wavevector sector
#                           The Navier-Stokes reference is integrated separately and cached, the plots dont change the calculation

#                           Start reading this file from: supported_history

from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply

from carleman.analysis.tse import supported_problem
from carleman.problems.navier_stokes import solve_primitive_navier_stokes_reference
from carleman.problems.fourier_vorticity import _validated_times


def evolve(matrix,initial,times):
    times=_validated_times(times)
    if len(times)==1:
        return initial[None,:].copy()
    # equally spaced times can share one exponential evaluation
    # the other branch always starts from initial, not from the previous snapshot
    intervals=np.diff(times)
    if np.allclose(intervals,intervals[0],rtol=1e-12,atol=1e-15):
        return expm_multiply(matrix,initial,start=0.,stop=float(times[-1]),num=len(times))
    return np.array([expm_multiply(time*matrix,initial) for time in times])


def reference_history(times, grid, epsilon, output, viscosity=.001):
    output=Path(output)
    path=output/f"NS_G{grid}_eps{epsilon:g}_nu{viscosity:g}_T{times[-1]:g}_M{len(times)}.npz"
    if not path.exists():
        reference=solve_primitive_navier_stokes_reference(times,viscosity=viscosity,grid_size=grid,epsilon=epsilon,
                                                        relative_tolerance=2e-12,absolute_tolerance=2e-14)
        output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,times=reference.times,modes=reference.modes,
                            velocity=reference.velocity_coefficients,omega=reference.vorticity_coefficients)
    with np.load(path) as saved:
        result={key:saved[key].copy() for key in saved.files}
    # the cache name only records the last time and count, check the actual times too
    np.testing.assert_allclose(result["times"],times,rtol=0,atol=1e-14)
    return result


def supported_history(problem,times,order):
    # the support was closed at the largest N, these lower-degree rows can contain extra zeros
    keep=problem["degrees"]<=order
    matrix=problem["matrix"][keep][:,keep]
    states=evolve(matrix,problem["initial"][keep],times)
    # slicing changes row numbers, map the original readout indices to the new rows
    positions=np.full(len(keep),-1,dtype=int)
    positions[keep]=np.arange(keep.sum())
    return states[:,positions[problem["vorticity_indices"]]]*problem["phase"]


def palinstrophy_history(order,radius,times,epsilon=Fraction(2),viscosity=.001):
    problem=supported_problem(order,radius,epsilon)
    positions={m:i for i,m in enumerate(problem["monomials"])}
    rates=np.array(problem["decay"])
    rows,columns=list(range(len(rates))),list(range(len(rates)))
    values=(-viscosity*rates).tolist()
    for source,monomial in enumerate(problem["monomials"]):
        for target,coupling in problem["transfers"][monomial].items():
            rows.append(positions[target]);columns.append(source);values.append(float(coupling))
    matrix=coo_matrix((values,(rows,columns)),shape=(len(rates),)*2).tocsr()
    initial=np.array(problem["initial"],dtype=float)
    # read P from the degree-two products, not by squaring the truncated vorticity
    readout=np.pi**2*np.array(problem["output"],dtype=float)
    histories=[]
    for cutoff in range(2,order+1):
        keep=np.array(problem["degrees"])<=cutoff
        states=evolve(matrix[keep][:,keep],initial[keep],times)
        histories.append(states@readout[keep])
    return np.array(histories)


def kinetic_energy_table(output,grid=144):
    times=np.array([0.,.1,.5,1.])
    rows=[]
    for epsilon in (.1,.5,1.,2.,5.):
        reference=reference_history(times,grid,epsilon,output)
        modes,velocity=reference["modes"],reference["velocity"]
        # the article measures the energy in the original oblique pair, not the difference from an unperturbed trajectory
        oblique=((modes[:,0]==2)&(modes[:,1]==1))|((modes[:,0]==-2)&(modes[:,1]==-1))
        share=100*(abs(velocity[:,oblique])**2).sum(axis=(-2,-1))/(abs(velocity)**2).sum(axis=(-2,-1))
        rows.append([epsilon,*share])
    return rows
