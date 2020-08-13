
from fourier_accountant.experimental.subsampled_gaussian_bounds import get_delta_max, get_delta_min

ncomp = 1000  # number of compositions of DP queries over minibatches
q     = 0.01  # subsampling ratio of minibatch
sigma = 4.0   # noise level for each query

upper_bound = get_delta_max(target_eps=1.0, sigma=sigma, q=q, ncomp=ncomp)
lower_bound = get_delta_min(target_eps=1.0, sigma=sigma, q=q, ncomp=ncomp)
print("the delta privacy parameter is bounded by ({}, {})".format(lower_bound, upper_bound))
