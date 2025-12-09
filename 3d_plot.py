"""
Trying to produce plots like Weng 2024 paper.
Usage:
    3d_plot.py <files>... --output=<dir>

Options:
    --output=<dir>   Directory to save plots.
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from dedalus.extras import plot_tools
from scipy.ndimage import gaussian_filter


def main(filename, start, count, output):
    """Save 3D plots for |psi_1| and |psi_2| for given range of analysis writes."""

    # Plot settings
    tasks = ['psi_1', 'psi_2']

    scale = 1.5
    dpi = 200
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)

    def abs_func(xmesh, ymesh, data):
        print(xmesh.shape)
        print(ymesh.shape)
        print(data.shape)
        print(data[0, 0])
        return xmesh, ymesh, np.abs(data)

    # Layout
    nrows, ncols = 2, 1
    image = plot_tools.Box(1, 1)
    pad = plot_tools.Frame(0.3, 0.1, 0.1, 0.1)
    margin = plot_tools.Frame(0.5, 0.3, 0.3, 0)

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure

    # Plot writes
    with h5py.File(filename, mode='r') as file:
        # open grids - assuming in same directory
        snapshots_dir = pathlib.Path(filename).parent
        grids_path = snapshots_dir / 'grids.h5'
        
        # open grids
        with h5py.File(grids_path, mode='r') as gfile:
            xmesh = gfile['x'][:]
            ymesh = gfile['y'][:]

        
        for index in range(start, start+count):
            for n, task in enumerate(tasks):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                axes = mfig.add_axes(i, j, [0, 0, 1, 1], projection='3d')
                # Call 3D plotting helper, slicing in time
                dset = file['tasks'][task]
                # 3D plot - slice along time axis, axis=0, pick time index 'index'
                
                # slice at given time index gives 2D field
                data_xy = dset[index,:,:]
                Xmesh, Ymesh, Z = abs_func(xmesh,ymesh,data_xy)
                
                # plot a smooth version
                Zsmooth = gaussian_filter(Z,sigma=5.0) # if really noisy increase sigma

                # need to fix the axes limits
                axes.set_zlim(0, 1.5) # this number is just based on the max from simulations; should be changed accordingly

                # set up plot
                ls = LightSource(315,45)

                # override built in shading
                rgb = ls.shade(Zsmooth, cmap= matplotlib.cm.gist_earth, vert_exag=0.1, blend_mode='soft')
                surf = axes.plot_surface(Xmesh, Ymesh, Z, rstride=1, cstride=1, facecolors=rgb, linewidth=0, antialiased=False, shade=False)
                
                cset = axes.contourf(
                        Xmesh, Ymesh, Zsmooth,
                        zdir='z', offset=0.0,
                        cmap='viridis'
                    )
                
                axes.set_title(f'$\{task}$')
                axes.set_xlabel(r"$x$")
                axes.set_ylabel(r"$y$")
                axes.set_zlabel(r"|$\psi$|")
                axes.view_init(elev=20, azim=45)

            # Add time title
            title = title_func(file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.44, y=title_height, ha='left')

            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
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
    post.visit_writes(args['<files>'], main, output=output_path)