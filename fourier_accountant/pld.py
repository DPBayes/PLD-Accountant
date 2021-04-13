from abc import ABCMeta, abstractmethod, abstractproperty
import typing
import numpy as np
import scipy.special
import scipy.optimize

class PrivacyLossDistribution(metaclass=ABCMeta):
    """ The distribution of the privacy loss resulting from application
    of a differentially private mechanism.
    """

    @abstractmethod
    def get_accountant_parameters(self, error_tolerance: float) -> typing.Tuple[float, float, int]:
        """ Determines suitable hyperparameters for the Fourier accountant
        for a given error tolerance.

        Args:
            - error_tolerance (float): The tolerance for error in approximations
                of bounds for delta.

        Returns:
            - L: Bound for the privacy loss interval to evaluate.
            - lambd: Parameter lambda for error bound computation.
            - nx: Number of discretization bins.
        """

    @abstractmethod
    def discretize_privacy_loss_distribution(self,
            start: float, stop: float, num_discretisation_bins_half: int
        ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Computes the privacy loss probability mass function evaluated for
        equally-sized discrete bins.

        Args:
            - start: Starting value for discretisation interval in privacy loss domain.
            - stop: (Exclusive) end value for discretisation interval in privacy loss domain.
            - num_discretisation_bins_half: The number of discretisation bins in the
                interval, divided by 2.

        Returns:
            - omega_y_L: np.ndarray of size `number_of_discretisation_bins`
                containing omega values for lower bound of delta.
            - omega_y_R: np.ndarray of size `number_of_discretisation_bins`
                containing omega values for upper bound of delta.
            - Lx: np.ndarray of size `number_of_discretisation_bins`
                containing the lower bound of the privacy loss intervals.
        """

class DiscretePrivacyLossDistribution(PrivacyLossDistribution):
    """ The privacy loss distribution defined by two discrete probability mass functions. """

    def __init__(self, p1: typing.Sequence[float], p2: typing.Sequence[float]) -> None:
        """
        Creates a new instance of DiscretePrivacyLossDistribution given
        probability mass functions represented as probability vectors `p1` and `p2`.

        It is required that values in `p1` and `p2` correspond to the same event/outcome,
        i.e., if `p1[i]` gives probability for some event `x` according to the first
        distribution, `p2[i]` must give the probability for the same event.

        Args:
            - p1: Sequence of probabilities expressing the probability mass
                function of the first distribution.
            - p2: Sequence of probabilities expressing the probability mass
                function of the second distribution.
        """
        if np.size(p1) != np.size(p2):
            raise ValueError("Both probability mass distributions must have the same size.")

        self._p1 = np.array(p1)
        self._p2 = np.array(p2)

    def get_accountant_parameters(self, error_tolerance: float) -> typing.Tuple[float, float, int]:
        raise NotImplementedError()

    @property
    def privacy_loss_values(self) -> np.ndarray:
        """ The values of the privacy loss random variable over the DP mechanisms output domain.

        Not ordered and may contain duplicates.
        """
        return np.log(self._p1 / self._p2)

    @property
    def privacy_loss_probabilities(self) -> np.ndarray:
        """ The probability mass omega associated with each privacy loss value. """
        return self._p1

    def discretize_privacy_loss_distribution(self,
            start: float, stop: float, num_discretisation_bins_half: int
        ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nx = 2 * num_discretisation_bins_half
        dx = (stop - start) / nx

        Lxs = self.privacy_loss_values
        ps = self.privacy_loss_probabilities

        omega_y_R = np.zeros(nx)
        iis = np.ceil((Lxs - start) / dx).astype(int)
        assert np.all((iis >= 0) & (iis < nx))
        np.add.at(omega_y_R, iis, ps)

        # note(lumip): the above is just a histogram computation, but currently does not fit neatly to np.histogram
        # because we are dealing with rightmost bin a bit oddly AND np.histogram includes rightmost border for some reason
        # which we (probably) don't want

        omega_y_L = np.zeros(nx)
        iis = np.floor((Lxs - start) / dx).astype(int)
        assert np.all((iis >= 0) & (iis < nx))
        np.add.at(omega_y_L, iis, ps)

        Xn = np.linspace(start, stop, nx, endpoint=False)
        return omega_y_L, omega_y_R, Xn

class ExponentialMechanism(DiscretePrivacyLossDistribution):
    """ The privacy loss distribution of the exponential mechanism
    where the quality score is a counting query.
    """

    def __init__(self, eps_em: float, m: int, n: int) -> None:
        """
        Creates an instance of the ExponentialMechanism.

        Args:
            eps_em: The epsilon value of the mechanism under composition.
            m: Number of elements accepted/counted by the query.
            n: Total number of elements in the counting query.
        """
        p1 = np.array([np.exp(eps_em*m),  np.exp(eps_em*(n-m))])
        p1 /= np.sum(p1)

        p2 = np.array([np.exp(eps_em*(m-1)),  np.exp(eps_em*(n-m))])
        p2 /= np.sum(p2)

        super().__init__(p1, p2)

    def get_accountant_parameters(self, error_tolerance: float) -> typing.Any:
        super().get_accountant_parameters(error_tolerance)

class SubsampledGaussianMechanism(PrivacyLossDistribution):
    """ The privacy loss distribution of the subsampled Gaussian mechanism
    with noise σ², subsampling ratio q.

    It is assumed that the provided noise level corresponds to a sensitivity
    of the mechanism of 1.
    """

    def __init__(self, sigma: float, q: typing.Optional[float] = 1.) -> None:
        """
        Args:
            sigma: Gaussian mechanism noise level for sensitivity 1.
            q: Subsampling ratio.
        """
        self.sigma = np.abs(sigma)
        self.q = q
        if self.q < 0 or self.q > 1:
            raise ValueError(f"Subsampling ratio q must be between 0 and 1, was {q}.")


    def get_accountant_parameters(self, error_tolerance: float) -> typing.Tuple[float, float, int]:
        raise NotImplementedError()

    def discretize_privacy_loss_distribution(self,
            start: float, stop: float, num_discretisation_bins_half: int
        ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nx = int(2 * num_discretisation_bins_half)

        Xn, dx = np.linspace(start, stop, nx+1, endpoint=True, retstep=True)
        assert dx == (stop - start) / nx

        fxs = self.evaluate(Xn) # privacy loss density evaluated at all intervals bounds, including both ends
        # interval i corresponds to bounds [i, i+1]

        # determine maximum value: since pld is unimodal,
        # must be in the interval left or right of the largest boundary value
        max_boundary_id = np.argmax(fxs)
        max_domain_ids = (
            np.maximum(0, max_boundary_id - 1),
            np.minimum(nx - 1, max_boundary_id + 1)
        )
        max_domain = (np.maximum(np.log(1 - self.q), Xn[max_domain_ids[0]]), Xn[max_domain_ids[1]])
        opt_result = scipy.optimize.minimize_scalar(
            lambda x: -self.evaluate(x), bounds=max_domain, method='bounded'
        )
        assert opt_result.success
        x_max = opt_result.x    # location of maximum privacy loss density value
        fx_max = -opt_result.fun # maximum of privacy loss density
        x_max_idx = int((x_max - start) // dx)
        assert x_max_idx >= 0 and x_max_idx < nx

        # Majorant for privacy loss density: Maximal value in the interval containing it
        # and the bound closer to maximum in all other intervals.
        omega_R = np.zeros(nx) # the max privacy loss density for each of the intervals;
        omega_R[x_max_idx]      = fx_max
        omega_R[:x_max_idx]     = fxs[1 : x_max_idx + 1]  # right boundaries for intervals before maximum
        omega_R[x_max_idx + 1:] = fxs[x_max_idx + 1 : -1] # left boundaries for intervals after maximum

        # Minorant for privacy loss density: smaller bound for interval containing the maximum
        # and the bound farther from maximum in all other intervals.
        omega_L = np.zeros(nx) # the min privacy loss density for each of the intervals
        omega_L[x_max_idx]      = np.min(fxs[x_max_idx:x_max_idx + 2])
        omega_L[:x_max_idx]     = fxs[:x_max_idx]     # left boundaries for intervals before maximum
        omega_L[x_max_idx + 1:] = fxs[x_max_idx + 2:] # right boundaries for intervals after maximum

        omega_R *= dx
        omega_L *= dx

        assert np.all(omega_R >= omega_L)

        return omega_L, omega_R, Xn[:-1]

    def _evaluate_internals(self,
            x: typing.Sequence[float], compute_derivative: typing.Optional[bool]=False
        ) -> typing.Union[np.array, typing.Tuple[np.array, np.array]]:
        """ Computes common values for PLD and its derivative.

        Args:
            - x: Privacy loss values for which to compute the probability density.
            - compute_derivative: If True, also outputs the derivative of the
                PLD evaluated at `x`.
        Returns:
            - omega: np.ndarray containing the probability density values of
                the PLD for each value in `x`
            - domega: (Only if `compute_derivative` is `True`): np.ndarray containing
                the values of the derivative of the probability density functions
                evaluated at each value in `x`.
        """
        sigma = self.sigma
        q = self.q

        mask = x > np.log(1 - q)

        sigma_sq = sigma**2
        exp_x = np.exp(x[mask])
        exp_x_m_1mq = exp_x - (1 - q)

        # g(s) in AISTATS2021 paper, Sec. 6.3
        Linvx = sigma_sq * ( np.log(exp_x_m_1mq) - np.log(q) ) + 0.5

        gauss_exp_term_1mq = np.exp(-Linvx**2 / (2 * sigma_sq))
        gauss_exp_term_q   = np.exp(-(Linvx-1)**2 / (2 * sigma_sq))

        # f(g(s)) in AISTATS2021 paper, Sec. 6.3
        ALinvx = ( 1/np.sqrt(2 * np.pi * sigma_sq) ) * (
            (1 - q) * gauss_exp_term_1mq + q  * gauss_exp_term_q
        )

        # g'(s)
        dLinvx = sigma_sq * exp_x /  exp_x_m_1mq

        omega = np.zeros_like(x)
        omega[mask] = ALinvx * dLinvx

        if not compute_derivative:
            return omega

        dALinvx = -( 1/(np.sqrt(2 * np.pi * sigma_sq) * sigma_sq) ) * (
            (1 - q) * gauss_exp_term_1mq * Linvx + q * gauss_exp_term_q * (Linvx - 1)
        )

        ddLinvx = sigma_sq * exp_x * (q - 1) / exp_x_m_1mq**2

        domega = np.zeros_like(x)
        domega[mask] = dALinvx * dLinvx**2 + ALinvx * ddLinvx
        return omega, domega

    def evaluate(self, x: typing.Sequence[float]) -> np.ndarray:
        """ Evaluates the probability densitiy function.

        Args:
            - x: Privacy loss values for which to compute the probability density.
        Returns:
            np.ndarray containing the probability density values of
                the PLD for each value in `x`
        """
        return self._evaluate_internals(x, compute_derivative=False)

    def evaluate_derivative(self, x: typing.Sequence[float]) -> np.ndarray:
        """ Evaluates the derivative of the probability densitiy function.

        Args:
            - x: Privacy loss values for which to compute the derivate of the
                probability density functions.
        Returns:
            np.ndarray containing the values of the derivative of the probability
                densitiy function evaluated each value in `x`
        """
        return self._evaluate_internals(x, compute_derivative=True)[1]


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

    # evaluate the error bound of Thm. 10
    if isinstance(pld, DiscretePrivacyLossDistribution):
        # if pld is a DiscretePrivacyLossDistribution we can get
        #  privacy loss values and corresponding probabilities directly for
        #  the error computation and don't need to rely on the discretisation
        #  (which we still need for the FFTs, however)
        Lxs = pld.privacy_loss_values
        ps = pld.privacy_loss_probabilities
    else:
        ps = omega_y # note(lumip): bounds probabilities from above (for truncated region),
                       # which seems more appropriate for the error term than bounding from below
                       # todo(all): verify this makes sense

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

    # evaluate the error bound of Thm. 10
    if isinstance(pld, DiscretePrivacyLossDistribution):
        # if pld is a DiscretePrivacyLossDistribution we can get
        #  privacy loss values and corresponding probabilities directly for
        #  the error computation and don't need to rely on the discretisation
        #  (which we still need for the FFTs, however)
        Lxs = pld.privacy_loss_values
        ps = pld.privacy_loss_probabilities
    else:
        ps = omega_y_R # note(lumip): bounds probabilities from above (for truncated region),
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

class PrivacyException(Exception):
    """ An exception indicating a violation of privacy constraints. """

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

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

    # evaluate the error bound of Thm. 10
    if isinstance(pld, DiscretePrivacyLossDistribution):
        # if pld is a DiscretePrivacyLossDistribution we can get
        #  privacy loss values and corresponding probabilities directly for
        #  the error computation and don't need to rely on the discretisation
        #  (which we still need for the FFTs, however)
        Lxs = pld.privacy_loss_values
        ps = pld.privacy_loss_probabilities
    else:
        ps = omega_y_R # note(lumip): bounds probabilities from above (for truncated region),
                       # which seems more appropriate for the error term than bounding from below
                       # todo(all): verify this makes sense

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

    # evaluate the error bound of Thm. 10
    if isinstance(pld, DiscretePrivacyLossDistribution):
        # if pld is a DiscretePrivacyLossDistribution we can get
        #  privacy loss values and corresponding probabilities directly for
        #  the error computation and don't need to rely on the discretisation
        #  (which we still need for the FFTs, however)
        Lxs = pld.privacy_loss_values
        ps = pld.privacy_loss_probabilities
    else:
        ps = omega_y_R # note(lumip): bounds probabilities from above (for truncated region),
                       # which seems more appropriate for the error term than bounding from below
                       # todo(all): verify this makes sense

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
    num_compositions = 1000
    print("### Exponential Mechanism")
    em_pld = ExponentialMechanism(.1, 7, 10)
    minitest(em_pld, target_delta=.5, num_compositions=num_compositions)

    print("### Subsampled Gaussian Mechanism")
    q = 0.01
    sigma = 2
    sgm_pld = SubsampledGaussianMechanism(sigma, q)
    minitest(sgm_pld, target_delta=.00001, num_compositions=num_compositions)
    minitest(sgm_pld, target_delta=0, num_compositions=num_compositions)
    # minitest(sgm_pld, target_delta=1.0, num_compositions=num_compositions) # don't get there, eps=0 -> target_delta < 1

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

