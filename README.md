# Running plotting script for a range of specified times

 Plotting for each timestep can take hours if the time step is very small 
(e.g., $dt=0.001$ to avoid numerical instability), so if you want to plot for multiple times 
but only at larger timesteps you can do this via a for loop in the shell.

### Example: plot at $T=0.0,0.5,...,50.0$
Run from unix shell (zsh or bash) from home directory

```zsh
for k in $(seq 0 100); do
  T=$(echo "$k * 0.5" | bc)
  python ./Programming/pde-course/plot_at_specifiedT.py \
  Desktop/outputs_neweqn/128_128_sig200_gauss_0.001/snapshots/snapshots_s1.h5 \
  --output=Desktop/outputs_neweqn/128_128_sig200_gauss_0.001/plots_Tseries \
  --time=$T \
  --plot_type='3d' \
  --zmax=1.5 \
  echo "k=$k T=$T" # sanity check
done
```
You can similarly plot for non-uniformly spaced out times by looping through an explicit list in the shell.
