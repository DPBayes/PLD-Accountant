# Fourier Accountant

Python code for computing tight differential privacy guarantees for discrete mechanisms and the subsampled Gaussian mechanism.

The method is described in

[1]: Antti Koskela, Joonas Jälkö, Antti Honkela:
"Computing Tight Differential Privacy Guarantees Using FFT",
International Conference on Artificial Intelligence and Statistics (2020)

[2]: Antti Koskela, Joonas Jälkö, Lukas Prediger, Antti Honkela:
"Tight Differential Privacy for Discrete-Valued Mechanisms and for the Subsampled Gaussian Mechanism Using FFT",
International Conference on Artificial Intelligence and Statistics (2021)

[3]: Antti Koskela, Antti Honkela:
"Computing Differential Privacy Guarantees for Heterogeneous Compositions Using FFT",
arXiv preprint arXiv:2102.12412


# API and Usage

- `pld`: The `pld` module contains implementations of privacy loss distributions for specific privacy mechanisms.
  - `pld.PrivacyLossDistribution`: Abstract base class defining a common interface for privacy loss distributions.
  - `pld.DiscreteMechanism`: Used for representing PLDs of arbitrary discrete mechanisms for which probability masses can be provided completely by finite arrays.
  - `pld.ExponentialMechanism`: Used for representing PLDs of exponential mechanisms for a counting query (e.g. [2, Section 6.1]).
  - `pld.SubsampledGaussian`: Used for representing PLDs of the subsampled Gaussian mechanism as used for example in DP-SGD.
- `accountant`: The `accountant` module contains the implementation of the Fourier accountant for composition of mechanisms described by the above PLD implementations.
  - `accountant.get_delta_upper_bound`: Computes the upper bound for privacy parameter δ.
  - `accountant.get_delta_lower_bound`: Computes the lower bound for privacy parameter δ.
  - `accountant.get_epsilon_upper_bound`: Computes the upper bound for privacy parameter ε.
  - `accountant.get_epsilon_lower_bound`: Computes the lower bound for privacy parameter ε.

Outdated API for subsampled Gaussian only:
- `get_delta_R(target_eps, sigma, q, ncomp, nx, L)`
    Computes the DP delta for the remove/add neighbouring relation of datasets.
- `get_delta_S(target_eps, sigma, q, ncomp, nx, L)`
    Computes the DP delta for the substitute neighbouring relation of datasets.
- `get_epsilon_R(target_delta, sigma, q, ncomp, nx, L)`
    Computes the DP epsilon for the remove/add neighbouring relation of datasets.
- `get_epsilon_S(target_delta, sigma, q, ncomp, nx, L)`
    Computes the DP epsilon for the substitute neighbouring relation of datasets.

The `experimental` module contains preliminary implementations for the above papers that
are not yet integrated into the main API.

## Important Parameters for the Accountant
- `target_eps` (`float`): Target ε to compute δ for.
- `target_delta` (`float`): Target δ to compute ε for.
- `L` (`float`):  Limit for the approximation of the privacy loss distribution integral.
- `num_compositions` (`int`): Number of compositions (=applications) of the given mechniasm.
- `num_discretisation_bins_half` (`int`): The number of discretisation bins used by the accountant in each half of the approximation interval.

The following currently only applies to the old API for subsampled Gaussian only:
For parameters `sigma`, `q` and `ncomp` either a single scalar or an array can be passed.
If a scalar is passed, the value will be re-interpreted as an array of length `1`. Each
function then computes the privacy values (`delta` or `epsilon`) resulting
from a composition of subsampled Gaussian mechanism with following parameters:
- `ncomp[0]` times noise level `sigma[0]` and subsampling rate `q[0]`
- `ncomp[1]` times noise level `sigma[1]` and subsampling rate `q[1]`
- etc.
for a total of `np.sum(ncomp)` operations.

An exception is raised if `sigma`, `q` and `ncomp` are found to not be of the
same length.


## Usage Notes

Note that the functions rely on numerical approximations, which are influenced
by choice of parameters `num_discretisation_bins_half` and `L`. Increasing `L` increases the range over
which the integral of the privacy loss distribution is approximated. `L` must be chosen
large enough to contain the true ε, otherwise a `ValueError` is raised.

## Usage Example

The example below illustrates usage of the package to compute upper boudns for
privacy parameters for the repeated application of the subsampled Gaussian mechanism
in differentially private SGD with a fixed noise level σ².

```python
import fourier_accountant
import numpy as np

ncomp = 10000 # number of compositions of DP queries over minibatches = number of iterations of SGD
q     = 0.01  # subsampling ratio of minibatch
sigma = 4.0   # noise level for each query

# computing privacy parameters for given parameters in remove/add neighboring relation with
#  poisson subsampling of minibatches
pld = fourier_accountant.plds.SubsampledGaussianMechanism(
    sigma, q, fourier_accountant.plds.NeighborRelation.REMOVE_POISSON
)

# computing delta bounds for given epsilon
target_eps = 1.0
delta_upper = fourier_accountant.get_delta_upper_bound(
    pld, target_eps, ncomp, L=20, num_discretisation_bins_half=int(1E7)
)
delta_lower = fourier_accountant.get_delta_lower_bound(
    pld, target_eps, ncomp, L=20, num_discretisation_bins_half=int(1E7)
)
print(delta_lower, delta_upper)
# 1.2282282018518088e-07 0.00010514221537608886

# computing epsilon bounds for given delta
target_delta = 1e-5
eps_upper, _ = fourier_accountant.get_epsilon_upper_bound(
    pld, target_delta, ncomp, L=20, num_discretisation_bins_half=int(1E7)
)
eps_lower, _ = fourier_accountant.get_epsilon_lower_bound(
    pld, target_delta, ncomp, L=20, num_discretisation_bins_half=int(1E7)
)
print(eps_lower, eps_upper)
# 0.6980780002786826 1.1339061240664539
```

## PyPI package

The accountant can be downloaded as a PyPI package 'fourier-accountant':

```pip3 install fourier-accountant```
