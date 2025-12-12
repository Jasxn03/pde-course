"""
Dedalus code for 2.11 of Darryl's notes
To run in parallel do mpirun -np [# processes] python3 [this_file_name.py]
"""

OMP_NUM_THREADS = 1

from dedalus import public as de
import numpy as np
import h5py, os
np.seterr(all='raise')


# parameters - we want to derive a CFL condition
Lx, Ly = 1000, 1000
Nx, Ny = 128, 128
F = 1 # do not change - set by problem
alpha_sq = 1 # do not change
dt = 0.05
t_end = 50
nout = 200
A0 = 1
noise_amp = 1e-3
bathymetry_type = "ridge" #can use flat, gauss, ridge, random


# OUTPUT - CHANGE TO YOUR FOLDER
outdir = f'/Users/ntj21/Desktop/outputs/{Nx}_{Ny}_{bathymetry_type}_{dt}/snapshots'
os.makedirs(outdir, exist_ok=True)


# domain
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

X,Y = np.meshgrid(np.linspace(0,Lx, Nx, endpoint=False),np.linspace(0,Ly, Ny, endpoint=False),indexing='ij')

# be parallel safe :)
x_local = dist.local_grid(x_basis)
y_local = dist.local_grid(y_basis)

# needed to bring initial conditions before solving
rng = np.random.default_rng(1234)

gshape = psi_1['g'].shape
gshape_2 = psi_2['g'].shape
D0_shape = D0_field['g'].shape
print(D0_shape)

noise = noise_amp*(rng.standard_normal(gshape) + 1j*rng.standard_normal(gshape))
noise_2 = noise_amp*(rng.standard_normal(gshape_2) + 1j*rng.standard_normal(gshape_2))
psi_1['g'] = A0*(1 + noise)
psi_2['g'] = A0*(1 + noise_2)

# bathymetry
Dbar = 1.0

if bathymetry_type == 'flat':
    D0 = Dbar * np.ones([D0_shape[0], D0_shape[1]])
elif bathymetry_type == 'gauss':
    A_bump, sigma = 2.0, 200.0 # originally sigma was 5, changed so bump isnt so narrow, gets rid of underflow error
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

# problem
F_val = F
problem = de.IVP([psi_1, psi_2, psi_1_star, psi_2_star, J1, J2],
                 time='t', namespace={"D0_field": D0_field,
                                      "F_val": F_val,
                                      "a2": alpha_sq,
                                      "eps": 1e-10})


# first coupled equation - using 2.16 in Darryl's notes!!!
problem.add_equation(("1j * dt(psi_1) = 1/abs(psi_1)**2 * dot((J1 + J2), grad(psi_1))" +
                      "- a2/2 * lap(sqrt(abs(psi_1_star * psi_1) + eps)) * psi_1 / (sqrt(psi_1_star * psi_1 + eps))" + 
                      "+ abs(psi_2)**2/(2 * F_val) * psi_1"))
# second coupled equation
problem.add_equation(("1j * dt(psi_2) = 1/abs(psi_1)**2 * dot((J1 + J2), grad(psi_2))" +
                      "+ psi_2 /(2 * F_val) * (abs(psi_1)**2 - 2 * D0_field)"))

# ensure jacobians etc are calculated correctly
problem.add_equation("J1 = 1/2j * (psi_1_star * grad(psi_1) - conj(psi_1_star * grad(psi_1)))")
problem.add_equation("J2 = 1/2j * (psi_2_star * grad(psi_2) - conj(psi_2_star * grad(psi_2)))")
problem.add_equation("psi_1_star = abs(psi_1)**2/psi_1")
problem.add_equation("psi_2_star = abs(psi_2)**2/psi_2")

solver = problem.build_solver(de.RK443)
print("solver built")

snapshots = solver.evaluator.add_file_handler(outdir, iter=10)
snapshots.add_task(psi_1, name="psi_1")
snapshots.add_task(psi_2, name="psi_2")
snapshots.add_task(D0_field, name="D0")

if dist.comm.rank == 0:
    with h5py.File(os.path.join(outdir, 'grids.h5'),'w') as f:
        f['x'] = X; f['y'] = Y

t, step = 0.0, 0
while solver.proceed and t < t_end:
    solver.step(dt)
    step += 1
    t += dt
    print(f"output snapshot {step} t={t:.3f}, max psi_1 = {np.max(abs(psi_1['g']))}, max psi_2 = {np.max(abs(psi_2['g']))}")
print('simulation finished')