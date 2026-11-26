# Inputs

As an input to our model we have FCN3 data for 4 ensemble members at t=6 over a region, and the ERA5 state at t=0. Likely we should average the 4 member samples, as this gives us permutation invariance. There is no meaning in any one member coming before or after another member in our input, since they should be i.i.d. samples from FCN3.

Our averaged FCN3 members will have the following set of features:

- u, v, z and t (wind components, geopotential and temperature)
- at FCN3's 5 pressure levels: 50, 100, 150, 200, 250 hPa
- so 4 × 5 = **20 features** per gridpoint

Alongside these we have the t=0 state: ERA5 u and v at all 10 levels (20), plus z and t at the 5 FCN3 levels (10). We also add 3 position features (sin lat, sin lon, cos lon), since a neighborhood on its own doesn't know where it is. That gives 20 + 20 + 10 + 3 = **53 features per gridpoint**. If we also want the ensemble spread, we can add the member std next to the mean (another 20 features, 73 total).

Not only will we want to use members for one position, we will want to use members for several positions in a neighborhood. This is because winds around the location of interest are likely to impact our prediction in future time steps. The neighborhood size can be a hyperparameter. Across an NxN neighborhood, with averaged FCN3 members, we will have 53 · N² total inputs (e.g. 53 · 64² ≈ 217k for a 64x64 neighborhood).

NOTE: Should we get more output from FCN3 we are missing things like humidity or sea level pressure that could be helpful in training.
 

# Outputs

Outputs of the model will be the u and v wind velocities for the hourly time steps following our input (t=0). That is, we want the u and v velocities at time steps t=1,2,3,4,5,6 for 7 pressure levels: 30, 50, 70, 100, 125, 150 and 175 hPa. This is because these are the levels weather balloons typically fly at.

Only 50, 100 and 150 hPa are FCN3 levels. The model has to fill in 30, 70, 125 and 175 hPa from the levels around them (and 30 hPa is above FCN3's top level), so we should expect larger errors there.

All together we will have 6 * 7 * 2 = 84 outputs per gridpoint. Depending on the architecture we try, these could come one at a time or be output all together.

NOTE: We can also output an ensemble, getting one prediction per FCN3 input, and skip averaging them to begin with. we can then average our outputs rather than the inputs.
