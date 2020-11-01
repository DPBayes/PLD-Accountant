# Fourier Accountant

Python code for computing tight DP-guarantees for the subsampled Gaussian mechanism.

The method is described in:

Antti Koskela, Joonas Jälkö, Antti Honkela:  
Computing Tight Differential Privacy Guarantees Using FFT  
International Conference on Artificial Intelligence and Statistics (2020)

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
- `sigma` (`float` or `np.ndarray`): Privacy noise sigma values
- `q` (`float` or `np.ndarray`): Subsampling ratios, i.e., how large are batches relative to the dataset
- `ncomp` (`int` or `np.ndarray` with `integer` type): Number of compositions, i.e., how many subsequent batch operations are queried
- `nx` (`int`): Number of discretiation points
- `L` (float):  Limit for the approximation of the privacy loss distribution integral

For parameters `sigma`, `q` and `ncomp` either a single scalar or an array can be passed.
If a scalar is passed, the value will be re-interpreted as an array of length `1`. Each
function then computes the privacy values (`delta` or `epsilon`) resulting
from a composition of subsampled Gaussian mechanism with following parameters:
- `ncomp[0]` times noise level `sigma[0]` and subsamplign rate `q[0]`
- `ncomp[1]` times noise level `sigma[1]` and subsamplign rate `q[1]`
- etc.
for a total of `np.sum(ncomp)` operations.

An exception is raised if `sigma`, `q` and `ncomp` are found to not be of the
same length.


## Usage Notes

Note that the functions rely on numerical approximations, which are influenced
by choice of parameters `nx` and `L`. Increasing `L` increases the range over
which the integral of the privacy loss distribution is approximated. `L` must be chosen
large enough, otherwise a `ValueError` is raised (in `get_epsilon_*`).
`nx` is the number of evaluation points in $[-L,L]$.

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

# computing delta for given epsilon for remove/add neighbouring relation
# with varying parameters
ncomp = np.array([500, 500])
q     = np.array([0.01, 0.01])
sigma = np.array([2.0, 1.0])
delta = fourier_accountant.get_delta_R(target_eps=1.0, sigma=sigma, q=q, ncomp=ncomp)
print(delta)
# 0.0003151995621652058
```
