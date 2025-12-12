from dedalus import public as de
import numpy as np
import logging
import h5py
import os

log = logging.getLogger("solver")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

np.seterr(all='raise')  # terminate when overflow/underflow occurs

"""
Solve the coupled equations for buoyancy using Dedalus
Parameters
----------
- Lx, Ly: length of the x and y components of the domain (starting from 0)
- Nx, Ny: number of gridpoints in the x and y directions
- dt: time step
- t_end: end time of the simulation
- n_out: record snapshot of psi every n_out * dt seconds
- A0: initial condition value
- noise_amp: add noise to the initial condition, parameter controls amplitude
- bathymetry_type: type of bathymetry considered.
    see below for which types are implemented
"""

# parameters
Lx, Ly = 1000, 1000
Nx, Ny = 128, 128
dt = 0.05
t_end = 50
n_out = 200
A0 = 1
noise_amp = 1e-3

"Bathymetry options: {'flat', 'gauss', 'ridge', 'random'}"
bathymetry_type = "ridge"


# make output directory - change to own folder
outdir = f'/Users/ntj21/Desktop/outputs/{Nx}_{Ny}_{bathymetry_type}_{dt}/snapshots'
os.makedirs(outdir, exist_ok=True)


# set up the domain and necessary fields
coords = de.CartesianCoordinates('x', 'y')

x_basis = de.Fourier(coords['x'], Nx, bounds = (0,Lx), dtype = np.complex128)
y_basis = de.Fourier(coords['y'], Ny, bounds = (0, Ly), dtype = np.complex128)

dist = de.Distributor(coords, dtype=np.complex128)

psi_1 = dist.Field(name='psi_1', bases=(x_basis, y_basis))
psi_1_star = dist.Field(name="psi_1_star", bases=(x_basis, y_basis))

psi_2 = dist.Field(name="psi_2", bases=(x_basis, y_basis))
psi_2_star = dist.Field(name="psi_2_star", bases=(x_basis, y_basis))

J1 = dist.VectorField(coords, name="J1", bases=(x_basis, y_basis))
J2 = dist.VectorField(coords, name="J2", bases=(x_basis, y_basis))

D0_field = dist.Field(name="D0_field", bases=(x_basis, y_basis))

X, Y = np.meshgrid(np.linspace(0, Lx, Nx, endpoint=False),
                   np.linspace(0, Ly, Ny, endpoint=False),
                   indexing='ij')

# setting up initial conditions and bathymetry in a parallel safe manner
x_local = dist.local_grid(x_basis)
y_local = dist.local_grid(y_basis)


# initial conditions
gshape = psi_1['g'].shape
gshape_2 = psi_2['g'].shape
D0_shape = D0_field['g'].shape

rng = np.random.default_rng(1234)

noise = noise_amp*(rng.standard_normal(gshape) + 1j*rng.standard_normal(gshape))
noise_2 = noise_amp*(rng.standard_normal(gshape_2) + 1j*rng.standard_normal(gshape_2))
psi_1['g'] = A0*(1 + noise)
psi_2['g'] = A0*(1 + noise_2)

# bathymetry
Dbar = 1.0

if bathymetry_type == 'flat':
    D0 = Dbar * np.ones([D0_shape[0], D0_shape[1]])
elif bathymetry_type == 'gauss':
    A_bump, sigma = 2.0, 200.0
    x0, y0 = 0.5*Lx, 0.5*Ly
    D0 = Dbar + A_bump * np.exp(-((x_local-x0)**2 + (y_local-y0)**2)/(2*sigma**2))
elif bathymetry_type == 'ridge':
    m_ridge, A_ridge = 4, 0.5
    D0 = Dbar + A_ridge*np.cos(2*np.pi*m_ridge*x_local/Lx)
elif bathymetry_type == 'random':
    rng = np.random.default_rng(1)
    D0 = Dbar + 0.2*rng.standard_normal(D0_shape)
else:
    raise ValueError('Unknown bathymetry type')

D0_field['g'] = D0

# set up the problem
problem = de.IVP([psi_1, psi_2, psi_1_star, psi_2_star, J1, J2],
                 time='t', namespace={"D0_field": D0_field,
                                      "eps": 1e-10})

# add the first coupled equation
problem.add_equation(("1j * dt(psi_1) = 1/abs(psi_1)**2 * dot((J1 + J2), grad(psi_1))" +
                      "- 1/2 * lap(sqrt(abs(psi_1_star * psi_1) + eps)) * psi_1 / (sqrt(psi_1_star * psi_1 + eps))" +
                      "+ abs(psi_2)**2/(2) * psi_1"))
# add the second coupled equation
problem.add_equation(("1j * dt(psi_2) = 1/abs(psi_1)**2 * dot((J1 + J2), grad(psi_2))" +
                      "+ psi_2 /(2) * (abs(psi_1)**2 - 2 * D0_field)"))

# equations to ensure jacobians and conjugates are calculated correctly
problem.add_equation("J1 = 1/2j * (psi_1_star * grad(psi_1) - conj(psi_1_star * grad(psi_1)))")
problem.add_equation("J2 = 1/2j * (psi_2_star * grad(psi_2) - conj(psi_2_star * grad(psi_2)))")
problem.add_equation("psi_1_star = abs(psi_1)**2/psi_1")
problem.add_equation("psi_2_star = abs(psi_2)**2/psi_2")

solver = problem.build_solver(de.RK443)
log.info("solver built")

# choose what to output for plotting
snapshots = solver.evaluator.add_file_handler(outdir, iter=10)
snapshots.add_task(psi_1, name="psi_1")
snapshots.add_task(psi_2, name="psi_2")
snapshots.add_task(D0_field, name="D0")

if dist.comm.rank == 0:
    with h5py.File(os.path.join(outdir, 'grids.h5'), 'w') as f:
        f['x'] = X
        f['y'] = Y

# solve the problem
t, step = 0.0, 0
while solver.proceed and t < t_end:
    solver.step(dt)
    step += 1
    t += dt
    log.info((f"output snapshot {step} t={t:.3f}, " +
              f"max psi_1 = {np.max(abs(psi_1['g']))}, " +
              f"max psi_2 = {np.max(abs(psi_2['g']))}"))
log.info('simulation finished')
