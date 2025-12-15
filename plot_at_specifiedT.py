"""
Plot a single 3D or 2D snapshot at simulation time closest to time specified, T.
Usage:
    plot_at_T.py <files>... --output=<dir> --time=<T> [--plot_type=<type>] [--sigma=<s>] [--zmax=<z>]

Options:
    --output=<dir>   Directory to save plots.
    --time=<T> Simulation time to plot (float).
    --plot_type=<type> Type of plot 2D or 3D (default:2d).
    --sigma=<s> Gaussian smoothing sigma (default: 5.0)
    --zmax=<z> z-axis max (3d only, default: 1.1)
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from dedalus.extras import plot_tools
from scipy.ndimage import gaussian_filter
import pathlib

def main(filename, start, count, output, T=0.0, plot_type="2d", sigma=5.0, zmax=1.1):
    
    # plot settings
    tasks = ["psi_1", "psi_2"]
    scale = 1.5
    dpi=200
    
    # Plot type
    plot_type = str(plot_type).lower()
    if plot_type not in ('2d','3d'):
        raise ValueError(f"--type must be '2d' or '3d', {plot_type} is not implemented.")

    # Args
    T = float(T)
    sigma = float(sigma)
    zmax = float(zmax)

    # Plot layout
    nrows, ncols = 2, 1
    image = plot_tools.Box(1, 1)
    pad = plot_tools.Frame(0.3, 0.1, 0.1, 0.1)
    margin = plot_tools.Frame(0.5, 0.3, 0.3, 0)
    ls = LightSource(315,45)

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure
    
    
    # Plot writes
    # open grids - assuming in same directory
    snapshots_dir = pathlib.Path(filename).parent
    grids_dir = snapshots_dir / 'grids.h5'

    with h5py.File(grids_dir, mode='r') as gfile:
        xmesh = gfile['x'][:]
        ymesh = gfile['y'][:]

    # Define 3d and 2d plots
    def plot_3d(axes, Zsmooth):
        rgb = ls.shade(Zsmooth, cmap= matplotlib.cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        axes.plot_surface(xmesh, ymesh, Zsmooth, rstride=1, cstride=1, facecolors=rgb, linewidth=0, antialiased=False, shade=False)
        axes.contourf(
                        xmesh, ymesh, Zsmooth,
                        zdir='z', offset=0.0,
                        cmap='viridis'
                     ) 
        axes.set_zlim(0.0, zmax)
        # axes.set_title(f'$\{task}$')
        axes.set_xlabel(r"$x$")
        axes.set_ylabel(r"$y$")
        axes.set_zlabel(r"|$\psi$|")
        axes.view_init(elev=20, azim=45)
        

    def plot_2d(axes,Zsmooth):
         levels=50 # increase for smoother contours!
         plot = axes.contourf(
                        xmesh, ymesh, Zsmooth, levels=levels,
                        cmap='viridis'
                     ) 
         axes.set_xlabel(r"$x$")
         axes.set_ylabel(r"$y$")
         cb = fig.colorbar(plot, ax=axes, fraction=0.05, pad=0.04)
         cb.set_label(r'$|\psi|$')

    # open file and choose nearest time index
    with h5py.File(filename, mode='r') as file:
        sim_time = file["scales/sim_time"][:]
        idx = int(np.argmin(np.abs(sim_time - T)))
        t = float(file["scales/sim_time"][idx])
        write = int(file['scales/write_number'][idx])

        # build plots 
        for n, task in enumerate(tasks):
            i, j = divmod(n, ncols)

            if plot_type == '3d':
                axes = mfig.add_axes(i, j, [0, 0, 1, 1], projection='3d')
            else:
                axes = mfig.add_axes(i,j, [0,0,1,1])
            
            dset = file['tasks'][task]
            data = dset[idx,:,:]

            Z = np.abs(data)
            Zsmooth = gaussian_filter(Z,sigma=sigma)

            if plot_type == '3d':
                plot_3d(axes, Zsmooth)
            else:
                plot_2d(axes, Zsmooth)
            
            axes.set_title(f'$\{task}$')
        
        # Suptitle and save figure
        type_str = "3D surface " if plot_type == '3d' else '2D snapshot '
        title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
        fig.suptitle(
            f'{type_str} for t = {t:.3f} \n' f'(requested time: {T:.3f})',
            x=0.44, y=title_height, ha='left')

        # Save figure
        savename = f'{plot_type}_t{t:.3f}_write_{write:06d}.png'
        savepath = output.joinpath(savename)
        fig.savefig(str(savepath), dpi=dpi, bbox_inches="tight", pad_inches=0.1)
        fig.clear()
    plt.close(fig)

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    
    sigma = float(args["--sigma"]) if args.get("--sigma") is not None else 5.0
    zmax  = float(args["--zmax"])  if args.get("--zmax")  is not None else 1.1

    post.visit_writes(
        args['<files>'], 
        main, 
        output=output_path, 
        T=float(args["--time"]),
        plot_type=args["--plot_type"],
        sigma=sigma,
        zmax=zmax)