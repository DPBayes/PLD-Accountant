# Fourier Accountant

Python code for computing tight DP-guarantees for the subsampled Gaussian mechanism.

The method is described in:

Antti Koskela, Joonas Jälkö, Antti Honkela:
Computing Tight Differential Privacy Guarantees Using FFT

https://arxiv.org/abs/1906.03049

# API and Usage

- `get_delta_R(target_eps, sigma, q, ncomp, nx, L)`
    Computes the DP delta for the remove/add neighbouring relation of datasets.
- `get_delta_S(target_eps, sigma, q, ncomp, nx, L)`
    Computes the DP delta for the substitute neighbouring relation of datasets.
- `get_epsilon_R(target_delta, sigma, q, ncomp, nx, L)`
    Computes the DP epsilon for the remove/add neighbouring relation of datasets.
- `get_epsilon_S(target_delta, sigma, q, ncomp, nx, L)`
    Computes the DP epsilon for the substitute neighbouring relation of datasets.

## Parameters
- `target_eps` (`float`): Target epsilon to compute delta for
- `target_delta` (`float`): Target delta to compute epsilon for
- `sigma` (`float`): Privacy noise sigma
- `q` (`float`): Subsampling ratio, i.e., how large are batches relative to the dataset
- `ncomp` (`int`): Number of compositions, i.e., how many subsequent batch operations are queried
- `nx` (`int`): Number of discretiation points
- `L` (float):  Limit for the approximation of the privacy loss distribution integral

## Usage Notes

Note that the functions rely on numerical approximations, which are influenced
by choice of parameters `nx` and `L`. Increasing `L` roughly increases the range over
which the integral of the privacy loss distribution is approximated. `L` must be chosen
large enough to cover the computed epsilon, otherwise a `ValueError` is raised (in `get_epsilon_*`).
`nx` is the number of evaluation points in $[-L,L]$.
If you find results output by the functions to be inaccurate, try increasing these two parameters.

Due to numerical instabilities, corner cases exist where functions sometimes returns
inaccurate values. If you think this is occuring, increasing `nx` and verifying that
the returned value does not change by much is usually a good heuristic to verify the output.

## Usage Example

```python
import fourier_accountant

ncomp = 1000  # number of compositions of DP queries over minibatches
q     = 0.01  # subsampling ratio of minibatch
sigma = 4.0   # noise level for each query

# computing delta for given epsilon for remove/add neighbouring relation
delta = fourier_accountant.get_delta_R(target_eps=1.0, sigma=sigma, q=q, ncomp=ncomp)
print(delta)
# 4.243484012034273e-06

# computing epsilon for given delta for substitute neighbouring relation
eps = fourier_accountant.get_epsilon_S(target_delta=1e-5, sigma=sigma, q=q, ncomp=ncomp)
print(eps)
# 1.9931200626285734
```
