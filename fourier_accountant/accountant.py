import typing
import numpy as np
import scipy.special
from fourier_accountant.plds import PrivacyLossDistribution, PrivacyException, DiscretePrivacyLossDistribution

__all__ = ['get_delta_upper_bound', 'get_delta_lower_bound', 'get_epsilon_upper_bound', 'get_epsilon_lower_bound']

def _get_ps_and_Lxs(
        pld: PrivacyLossDistribution, omegas: np.ndarray, omega_Lxs: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
    """ Get the best representation of privacy loss probability mass for computing
        error term.

    For computation of the delta error term, these are not required to be discretised
    to regular intervals, so more efficient representations are possible.

    Args:
        - pld: Privacy loss distribution instance.
        - omegas: Discretized privacy loss probability masses.
        - omega_Lxs: Probability loss values corresponding to positions in `omegas`.

    Returns:
        - ps: Probability mass function for privacy loss values.
        - Lxs: The corresponding privacy loss values.
    """
    # todo(lumip): This should ideally be bundled in the error computation,
    #   but that would make that function's interface quite bloated, which indicates
    #   it should be part of PLD classes. However, that would in turn strongly
    #   couple those with the accountant computations - tricky...

    if isinstance(pld, DiscretePrivacyLossDistribution):
        # if pld is a DiscretePrivacyLossDistribution we can get
        #  privacy loss values and corresponding probabilities directly for
        #  the error computation and don't need to rely on the discretisation.
        # these will typically be smaller arrays and thus faster to compute on.
        Lxs = pld.privacy_loss_values
        ps = pld.privacy_loss_probabilities
    else:
        ps = omegas
        Lxs = omega_Lxs

    return ps, Lxs

def _get_delta_error_term(
        Lxs: typing.Sequence[float],
        ps: typing.Sequence[float],
        num_compositions: int = 500,
        L: float = 20.0,
        lambd: typing.Optional[float] = None
    ) -> float:
    """ Computes the total error term for δ computed by the Fourier accountant
    for repeated application of a privacy mechanism.

    The computation follows Theorem 7 in Koskela & Honkela, "Computing Differential Privacy for
    Heterogeneous Compositions Using FFT", 2021, arXiv preprint, https://arxiv.org/abs/2102.12412 .

    Args:
        - Lxs: Sequence of privacy loss values.
        - ps: Sequence of privacy loss probability masses.
        - num_compositions: The number of compositions (=applications) of the privacy mechanism.
        - L: The truncation threshold (in privacy loss space) used by the accountant.
        - lambd: The parameter λ for error estimation.
    """

    if lambd is None:
        lambd = .5 * L

    assert np.size(ps) == np.size(Lxs)
    nonzero_probability_filter = ~np.isclose(ps, 0)
    ps = ps[nonzero_probability_filter]
    Lxs = Lxs[nonzero_probability_filter]
    assert np.all(ps > 0)

    # Compute the lambda-divergence \alpha^+
    alpha_plus = scipy.special.logsumexp(np.log(ps) + lambd * Lxs)

    # Compute the lambda-divergence \alpha^-
    alpha_minus = scipy.special.logsumexp(np.log(ps) - lambd * Lxs)

    k = num_compositions

    common_factor_log = -(L * lambd + np.log1p(-np.exp(-2 * L * lambd)))

    T1_log = k * alpha_plus + common_factor_log
    T2_log = k * alpha_minus + common_factor_log

    T_max_log = np.maximum(T1_log, T2_log)

    error_term = np.exp(T_max_log) * (np.exp(T1_log - T_max_log) + np.exp(T2_log - T_max_log))

    return error_term

def _delta_fft_computations(omegas: np.ndarray, num_compositions: int) -> np.ndarray:
    """ Core computation of privacy loss distribution convolutions using FFT.

    Args:
        - omegas: Numpy array of probability masses omega for discrete bins of privacy loss values
            for a single invocation of a privacy mechanism.
        - num_compositions: The number of sequential invocations of the privacy mechanism.
    Returns:
        - Numpy array of probability masses for the discrete bins of privacy loss values
            after `num_compositions` sequential invocations of the privacy mechanisms
            characterized by `omegas`.
    """
    # Flip omegas, i.e. fx <- D(omega_y), the matrix D = [0 I;I 0]
    nx = len(omegas)
    assert nx % 2 == 0
    half = nx // 2
    fx = np.concatenate((omegas[half:], omegas[:half]))
    assert np.size(fx) == np.size(omegas)

    # Compute the DFT
    FF1 = np.fft.rfft(fx)

    # Take elementwise powers and compute the inverse DFT
    cfx = np.real(np.fft.irfft((FF1 ** num_compositions)))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    cfx = np.concatenate((cfx[half:], cfx[:half]))

    return cfx # todo(lumip): there are sometimes values < 0, all quite small, probably should be 0 but numerical precision strikes... problem?

def _compute_delta(
        convolved_omegas: np.ndarray, target_eps: float, L: float, compute_derivative: bool=False
    ) -> typing.Union[float, typing.Tuple[float, float]]:
    """ Compute delta from privacy loss probability masses.

    Args:
        - convolved_omegas: Numpy array of probability masses after convolving all
            privacy mechanism invocations.
        - target_eps: The targeted epsilon to compute delta for.
        - L: The bound for the discretisation interval.
        - compute_derivative: If True, additionally return the derivative of delta with
            respect to epsilon.

    Returns:
        - delta: The computed delta.
        - ddelta (Optional, if `compute_derivative = True`): The derivative of delta wrt epsilon.
    """
    nx = len(convolved_omegas)
    # Evaluate \delta(target_eps)
    x = np.linspace(-L, L, nx, endpoint=False) # grid for the numerical integration
    integral_mask = x > target_eps
    x = x[integral_mask]
    convolved_omegas = convolved_omegas[integral_mask]

    dexp_e = -np.exp(target_eps - x)
    exp_e = 1 + dexp_e
    assert np.all(exp_e > 0)

    integrand = exp_e * convolved_omegas
    assert np.all(~(integrand < 0 ) | np.isclose(integrand, 0)), "encountered negative values in pld after composition"

    delta = np.sum(integrand)

    if not compute_derivative:
        return delta

    dintegrand = dexp_e * convolved_omegas
    ddelta = np.sum(dintegrand)
    return delta, ddelta

def get_delta_upper_bound(
        pld: PrivacyLossDistribution,
        target_eps: float,
        num_compositions: int,
        num_discretisation_bins_half: int = int(1E6),
        L: float = 20.0
    ):
    """ Computes the upper bound for privacy parameter δ for repeated application
    of a privacy mechanism.

    The computation follows the Fourier accountant method described in Koskela et al.,
    "Tight Differential Privacy for Discrete-Valued Mechanisms and for the Subsampled
    Gaussian Mechanism Using FFT", Proceedings of The 24th International Conference
    on Artificial Intelligence and Statistics, PMLR 130:3358-3366, 2021.

    Args:
        - pld: The privacy loss distribution of a single application of the privacy mechanism.
        - target_eps: The privacy parameter ε for which to compute δ.
        - num_compositions: The number of compositions (=applications) of the privacy mechanism.
        - num_discretisation_bins_half: The number of discretisation bins used by the accountant, divided by 2.
        - L: The truncation threshold (in privacy loss space) used by the accountant.
    """
    # obtain discretized privacy loss densities
    _, omega_y, Lxs = pld.discretize_privacy_loss_distribution(-L, L, num_discretisation_bins_half)

    # compute delta
    convolved_omegas = _delta_fft_computations(omega_y, num_compositions)
    delta = _compute_delta(convolved_omegas, target_eps, L)

    ps, Lxs = _get_ps_and_Lxs(pld, omega_y, Lxs)

    error_term = _get_delta_error_term(Lxs, ps, num_compositions, L)
    delta += error_term

    return np.clip(delta, 0., 1.)

def get_delta_lower_bound(
        pld: PrivacyLossDistribution,
        target_eps: float,
        num_compositions: int,
        num_discretisation_bins_half: int = int(1E6),
        L: float = 20.0
    ):
    """ Computes the lower bound for privacy parameter δ for repeated application
    of a privacy mechanism.

    The computation follows the Fourier accountant method described in Koskela et al.,
    "Tight Differential Privacy for Discrete-Valued Mechanisms and for the Subsampled
    Gaussian Mechanism Using FFT", Proceedings of The 24th International Conference
    on Artificial Intelligence and Statistics, PMLR 130:3358-3366, 2021.

    Args:
        - pld: The privacy loss distribution of a single application of the privacy mechanism.
        - target_eps: The privacy parameter ε for which to compute δ.
        - num_compositions: The number of compositions (=applications) of the privacy mechanism.
        - num_discretisation_bins_half: The number of discretisation bins used by the accountant, divided by 2.
        - L: The truncation threshold (in privacy loss space) used by the accountant.
    """
    # obtain discretized privacy loss densities
    omega_y_L, omega_y_R, Lxs = pld.discretize_privacy_loss_distribution(-L, L, num_discretisation_bins_half)

    # compute delta
    convolved_omegas = _delta_fft_computations(omega_y_L, num_compositions)
    delta = _compute_delta(convolved_omegas, target_eps, L)

    ps, Lxs = _get_ps_and_Lxs(pld, omega_y_R, Lxs) # note(lumip): bounds probabilities from above (for truncated region),
                       # which seems more appropriate for the error term than bounding from below
                       # todo(all): verify this makes sense

    error_term = _get_delta_error_term(Lxs, ps, num_compositions, L)
    delta -= error_term

    return np.clip(delta, 0., 1.)

def _compute_epsilon(
        convolved_omegas: np.ndarray, target_delta: float, tol: float, error_term: float, L: float
    ) -> typing.Tuple[float, float]:
    """ Find epsilon using Newton iteration on delta computation for given probability masses.

    Args:
        - convolved_omegas: Numpy array of probability masses after convolving all
            privacy mechanism invocations.
        - target_delta: The targeted delta to compute epsilon for.
        - tol: Optimisation cutoff threshold for epsilon.
        - error_term: Delta error term.
        - L: The bound for the discretisation interval.

    Returns:
        - epsilon: The computed value for epsilon.
        - delta: The value of delta corresponding to epsilon. Might differ from
            `target_delta` if a suitable epsilon for `target_delta` cannot be found.
    """

    last_epsilon = -np.inf
    epsilon = 0
    delta, ddelta = _compute_delta(convolved_omegas, epsilon, L, compute_derivative=True)
    delta += error_term
    delta = np.clip(delta, 0., 1.)
    while np.abs(target_delta - delta) > tol and not np.isclose(epsilon, last_epsilon):
        f_e = delta - target_delta
        df_e = ddelta
        last_epsilon = epsilon
        epsilon = np.maximum(last_epsilon - f_e/df_e, 0)

        delta, ddelta = _compute_delta(convolved_omegas, epsilon, L, compute_derivative=True)
        delta += error_term
        delta = np.clip(delta, 0., 1.)

    return epsilon, delta

def get_epsilon_upper_bound(
        pld: PrivacyLossDistribution,
        target_delta: float,
        num_compositions: int,
        num_discretisation_bins_half: int = int(1E6),
        L: float = 20.0,
        tol: float = 1e-9
    ):
    """ Computes the upper bound for privacy parameter ε for repeated application
    of a privacy mechanism.

    The computation optimizes for ε iteratively using the Newton method on
    the Fourier accountant for computing an upper bound for δ.
    The accountant is described in Koskela et al.,
    "Tight Differential Privacy for Discrete-Valued Mechanisms and for the Subsampled
    Gaussian Mechanism Using FFT", Proceedings of The 24th International Conference
    on Artificial Intelligence and Statistics, PMLR 130:3358-3366, 2021.

    Args:
        - pld: The privacy loss distribution of a single application of the privacy mechanism.
        - target_delta: The privacy parameter δ for which to compute ε.
        - num_compositions: The number of compositions (=applications) of the privacy mechanism.
        - num_discretisation_bins_half: The number of discretisation bins used by the accountant, divided by 2.
        - L: The truncation threshold (in privacy loss space) used by the accountant.
        - tol: Error tolerance for ε.
    """
    # obtain discretized privacy loss densities
    omega_y_L, omega_y_R, Lxs = pld.discretize_privacy_loss_distribution(-L, L, num_discretisation_bins_half)

    # compute convolved omegas
    convolved_omegas = _delta_fft_computations(omega_y_R, num_compositions)

    ps, Lxs = _get_ps_and_Lxs(pld, omega_y_R, Lxs)
    error_term = _get_delta_error_term(Lxs, ps, num_compositions, L)

    epsilon, delta = _compute_epsilon(convolved_omegas, target_delta, tol, error_term, L)

    if epsilon > L: raise ValueError("The evaluation bound L for privacy loss is too small.")
    if delta > target_delta + tol: raise PrivacyException("Could not find an epsilon for the given target delta.")
    assert epsilon >= 0., "Computed negative epsilon!"

    return epsilon, delta

def get_epsilon_lower_bound(
        pld: PrivacyLossDistribution,
        target_delta: float,
        num_compositions: int,
        num_discretisation_bins_half: int = int(1E6),
        L: float = 20.0,
        tol: float = 1e-9
    ):
    """ Computes the lower bound for privacy parameter ε for repeated application
    of a privacy mechanism.

    The computation optimizes for ε iteratively using the Newton method on
    the Fourier accountant for computing a lower bound for δ.
    The accountant is described in Koskela et al.,
    "Tight Differential Privacy for Discrete-Valued Mechanisms and for the Subsampled
    Gaussian Mechanism Using FFT", Proceedings of The 24th International Conference
    on Artificial Intelligence and Statistics, PMLR 130:3358-3366, 2021.

    Args:
        - pld: The privacy loss distribution of a single application of the privacy mechanism.
        - target_delta: The privacy parameter δ for which to compute ε.
        - num_compositions: The number of compositions (=applications) of the privacy mechanism.
        - num_discretisation_bins_half: The number of discretisation bins used by the accountant, divided by 2.
        - L: The truncation threshold (in privacy loss space) used by the accountant.
        - tol: Error tolerance for ε.
    """
    # obtain discretized privacy loss densities
    omega_y_L, omega_y_R, Lxs = pld.discretize_privacy_loss_distribution(-L, L, num_discretisation_bins_half)

    # compute convolved omegas
    convolved_omegas = _delta_fft_computations(omega_y_L, num_compositions)

    ps, Lxs = _get_ps_and_Lxs(pld, omega_y_R, Lxs)

    error_term = _get_delta_error_term(Lxs, ps, num_compositions, L)

    epsilon, delta = _compute_epsilon(convolved_omegas, target_delta, tol, -error_term, L)

    if epsilon > L: raise ValueError("The evaluation bound L for privacy loss is too small.")
    if delta > target_delta + tol: raise PrivacyException(f"Could not find an epsilon for the given target delta {target_delta}.")
    assert epsilon >= 0., "Computed negative epsilon!"

    return epsilon, delta


def minitest(pld, target_delta, num_compositions):
    eps_R, delta_eps_R = get_epsilon_upper_bound(pld, target_delta, num_compositions)
    assert np.isclose(target_delta, delta_eps_R), f"get_epsilon_upper_bound did not achieve target_delta {target_delta}"
    delta_eps_R_check_R = get_delta_upper_bound(pld, eps_R, num_compositions)
    assert np.isclose(delta_eps_R_check_R, delta_eps_R), f"computing delta from eps_R did not result in target_delta {target_delta}"

    eps_L, delta_eps_L = get_epsilon_lower_bound(pld, target_delta, num_compositions)
    assert np.isclose(target_delta, delta_eps_R), f"get_epsilon_lower_bound did not achieve target_delta {target_delta}"
    delta_eps_L_check_L = get_delta_lower_bound(pld, eps_L, num_compositions)
    assert np.isclose(delta_eps_L_check_L, delta_eps_L), f"computing delta from eps_L did not result in target_delta {target_delta}"
    print(f"eps domain for target_delta {target_delta} is [{eps_L}, {eps_R}]")

if __name__ == '__main__':
    from fourier_accountant.plds import ExponentialMechanism, SubsampledGaussianMechanism, NeighborRelation
    num_compositions = 1000
    print("### Exponential Mechanism")
    em_pld = ExponentialMechanism(.1, 7, 10)
    minitest(em_pld, target_delta=.5, num_compositions=num_compositions)

    print("### Subsampled Gaussian Mechanism, remove relation")
    q = 0.01
    sigma = 2
    sgm_pld = SubsampledGaussianMechanism(sigma, q)
    t = np.linspace(-10, 10, 501, endpoint=True)

    minitest(sgm_pld, target_delta=.00001, num_compositions=num_compositions)
    minitest(sgm_pld, target_delta=0, num_compositions=num_compositions)
    minitest(sgm_pld, target_delta=.4, num_compositions=num_compositions)

    print("### Subsampled Gaussian Mechanism, substitute relation")
    sgm_sub_pld = SubsampledGaussianMechanism(sigma, q, NeighborRelation.SUBSTITUTE_NO_REPLACE)
    minitest(sgm_sub_pld, target_delta=0, num_compositions=num_compositions)
    # note(lumip): for 1000 comps, substitution gives very small delta for eps=0 already; how to test?

    # note(lumip): verification with existing experimental and older code
    print("### comparing code versions")
    from experimental.subsampled_gaussian_bounds import get_delta_max
    from compute_delta import get_delta_R, get_delta_S

    target_eps = 0.00001
    delta = get_delta_upper_bound(sgm_pld, target_eps, num_compositions=1)
    delta_experimental_code = get_delta_max(target_eps, ncomp=1)
    delta_old_code_R = get_delta_R(target_eps, ncomp=1)
    delta_old_code_S = get_delta_S(target_eps, ncomp=1)
    print(f"new code: {delta}")
    print(f"experimental code: {delta_experimental_code}")
    print(f"old code: R relation: {delta_old_code_R}, S relation: {delta_old_code_S}")
    assert np.isclose(delta_old_code_R, delta)

    print("all successful")

