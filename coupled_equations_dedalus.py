"""
Dedalus code for 2.16 of Darryl's notes
"""

from dedalus import public as de
import numpy as np
import h5py, os 
np.seterr(all='raise')

#parameters
Lx, Ly = 100.0, 100.0
Nx, Ny = 512, 512
F = 1
alpha_sq = 1
dt = 0.002
t_end = 5
nout = 200
A0 = 1
noise_amp = 1e-3
bathymetry_type = "flat" #can use flat, gauss, ridge, random

#domain
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

rng = np.random.default_rng(1234)
noise = noise_amp*(rng.standard_normal(X.shape)+1j*rng.standard_normal(X.shape))
psi_1['g'] = A0*(1+noise)
psi_2['g'] = A0*(1+noise)

#bathymetry
Dbar = 1.0

if bathymetry_type == 'flat':
    D0 = Dbar * np.ones_like(X)
elif bathymetry_type == 'gauss':
    A_bump, sigma = 2.0, 5.0
    x0, y0 = 0.5*Lx, 0.5*Ly
    D0 = Dbar + A_bump *np.exp(-((X-x0)**2 + (Y-y0)**2)/(2*sigma**2))
elif bathymetry_type == 'ridge':
    m_ridge, A_ridge = 4, 0.5
    D0 = Dbar + A_ridge*np.cos(2*np.pi*m_ridge*X/Lx)
elif bathymetry_type == 'random':
    rng = np.random.default_rng(1)
    D0 = Dbar + 0.2*rng.standard_normal(X.shape)
else:
    raise ValueError('Unknown bathymetry type')

D0_field['g'] = D0

#problem
F_val = F
problem = de.IVP([psi_1, psi_2, psi_1_star, psi_2_star, J1, J2],
                 time='t', namespace={"D0_field": D0_field,
                                      "F_val": F_val,
                                      "a2": alpha_sq,
                                      "eps": 1e-10})


# first coupled equation - using 2.16 in Darryl's notes!!!
problem.add_equation(("1j * dt(psi_1) = 1/abs(psi_1)**2 * dot((J1 + J2), grad(psi_1))" +
                      "- a2/2 * lap(sqrt(abs(psi_1_star * psi_1) + eps)) * psi_1 / (sqrt(abs(psi_1_star * psi_1) + eps))" +  
                      "+ 1/(2 * 1j * F_val) * (abs(psi_1)**2 * psi_1 - D0_field * psi_1)"))
# second coupled equation
problem.add_equation(("1j * dt(psi_2) = 1/abs(psi_1)**2 * dot((J1 + J2), grad(psi_2))" +
                      "- psi_2 /(abs(psi_2)**2 * 4 * 1j * F_val) * (abs(psi_1)**4 - 2 * D0_field * abs(psi_1)**2)"))

# ensure jacobians etc are calculated correctly
problem.add_equation("J1 = 1/2j * (psi_1_star * grad(psi_1) - conj(psi_1_star * grad(psi_1)))")
problem.add_equation("J2 = 1/2j * (psi_2_star * grad(psi_2) - conj(psi_2_star * grad(psi_2)))")
problem.add_equation("psi_1_star = abs(psi_1)**2/psi_1")
problem.add_equation("psi_2_star = abs(psi_2)**2/psi_2")

solver = problem.build_solver(de.RK443)
print("solver built")

#output
outdir = 'data/snapshots'
os.makedirs(outdir, exist_ok = True)
if dist.comm.rank == 0:
    with h5py.File(os.path.join(outdir, 'grids.h5'),'w') as f:
        f['x'] = X; f['y'] = Y

t, step = 0.0, 0
while t < t_end:
    solver.step(dt)
    t = solver.sim_time
    step += 1
    if step % nout == 0 or solver.stop_iteration:
        psi_1_g = psi_1['g'].copy()
        psi_2_g = psi_2['g'].copy()
        D0_g = D0_field['g']
        if dist.comm.rank == 0:
            fname = os.path.join(outdir, f'snap_{step:05d}.h5')
            with h5py.File(fname, 'w') as f:
                f['psi'] = psi_1_g
                f['D0'] = D0_g
                f.attrs['t'] = t
        print(f'output snapshot {step} t={t:.3f} max|psi_1|={np.abs(psi_1_g).max():.4f}')
        print(f'output snapshot {step} t={t:.3f} max|psi_2|={np.abs(psi_2_g).max():.4f}')

print('simulation finished')