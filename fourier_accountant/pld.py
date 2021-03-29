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
    def get_accountant_parameters(self, error_tolerance: float) -> typing.Any:
        """ Determines suitable hyperparameters for the Fourier accountant
        for a given error tolerance.
        """

    @abstractmethod
    def discretize_privacy_loss_distribution(self,
            start: float, stop: float, number_of_discretisation_bins: int
        ) -> np.ndarray:
        """ Computes the privacy loss probability mass function evaluated for
        equally-sized discrete bins.

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

    def get_accountant_parameters(self, error_tolerance: float) -> typing.Any:
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
            start: float, stop: float, number_of_discretisation_bins: int
        ) -> np.ndarray:
        nx = number_of_discretisation_bins
        dx = (stop - start) / nx

        Lx = self.privacy_loss_values
        ps = self.privacy_loss_probabilities

        omega_y_R = np.zeros(nx)
        iis = np.ceil((Lx - start) / dx).astype(int)
        assert np.all((iis >= 0) & (iis < nx))
        np.add.at(omega_y_R, iis, ps)

        # note(lumip): the above is just a histogram computation, but currently does not fit neatly to np.histogram
        # because we are dealing with rightmost bin a bit oddly AND np.histogram includes rightmost border for some reason
        # which we (probably) don't want

        omega_y_L = np.zeros(nx)
        iis = np.floor((Lx - start) / dx).astype(int)
        assert np.all((iis >= 0) & (iis < nx))
        np.add.at(omega_y_L, iis, ps)

        Xn = np.linspace(start, stop, nx, endpoint=False)
        return omega_y_L, omega_y_R, Xn

class ExponentialMechanismPrivacyLossDistribution(DiscretePrivacyLossDistribution):
    """ The privacy loss distribution of the exponential mechanism
    where the quality score is a counting query.
    """

    def __init__(self, eps_em: float, m: int, n: int) -> None:
        """
        Creates an instance of the ExponentialMechanismPrivacyLossDistribution.

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

class SubsampledGaussianMechanismPrivacyLossDistribution(PrivacyLossDistribution):
    """ The privacy loss distribution of the subsampled Gaussian mechanism
    with noise σ², subsampling ratio q.

    It is assumed that the provided noise level correspond to a sensitivity
    of the mechanism of 1.
    """

    def __init__(self, sigma: float, q: typing.Optional[float] = 1.) -> None:
        """
        Args:
            sigma: Gaussian mechanism noise level for sensitivity 1.
            q: Subsampling ratio.
        """
        self.sigma = sigma
        self.q = q

    def get_accountant_parameters(self, error_tolerance: float) -> typing.Any:
        """ Determines suitable hyperparameters for the Fourier accountant for a given error tolerance. """
        raise NotImplementedError()

    def discretize_privacy_loss_distribution(self,
            start: float, stop: float, number_of_discretisation_bins: int
        ) -> np.ndarray:
        nx = int(number_of_discretisation_bins)

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
        ) -> np.ndarray:
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


def get_delta_error_term(
        pld: PrivacyLossDistribution,
        num_compositions: int = 500,
        L: float = 20.0,
        num_discretisation_points: int = int(1E6),
        lambd: typing.Optional[float] = None
    ):
    """ Computes the total error term for δ computed by the Fourier accountant
    for repeated application of a privacy mechanism.

    The computation follows Theorem 10 in Koskela et al., "Tight Differential Privacy
    for Discrete-Valued Mechanisms and for the Subsampled Gaussian Mechanism Using FFT",
    Proceedings of The 24th International Conference on Artificial Intelligence and Statistics,
    PMLR 130:3358-3366, 2021.

    Args:
        - pld: The privacy loss distribution of a single application of the privacy mechanism.
        - num_compositions: The number of compositions (=applications) of the privacy mechanism.
        - L: The truncation threshold (in privacy loss space) used by the accountant.
        - num_discretisation_points: The number of discretisation points used by the accountant.
        - lambd: The parameter λ for error estimation.
    """

    if lambd is None:
        lambd = .5 * L

    # Determine the privacy loss values and probabilities
    if isinstance(pld, DiscretePrivacyLossDistribution):
        Lx = pld.privacy_loss_values
        ps = pld.privacy_loss_probabilities
    else:
        omega_L, omega_R, Lx = pld.discretize_privacy_loss_distribution(-L, L, num_discretisation_points)
        ps = omega_R

    assert np.size(ps) == np.size(Lx)
    nonzero_probability_filter = ~np.isclose(ps, 0)
    ps = ps[nonzero_probability_filter]
    Lx = Lx[nonzero_probability_filter]

    # Compute the lambda-divergence \alpha^+
    alpha_plus = scipy.special.logsumexp(np.log(ps) + lambd * Lx)

    # Compute the lambda-divergence \alpha^-
    alpha_minus = scipy.special.logsumexp(np.log(ps) - lambd * Lx)

    # Evaluate the bound of Thm. 10
    k = num_compositions

    common_factor_log = -(L * lambd + np.log1p(-np.exp(-2 * L * lambd)))

    T1_log_num = (k+1) * alpha_plus + np.log(2) + np.log1p(-.5*(np.exp(-alpha_plus) + np.exp(-k * alpha_plus)))
    T1_log_denom = alpha_plus + np.log1p(-np.exp(-alpha_plus))

    T1_log = T1_log_num - T1_log_denom + common_factor_log

    T2_log_num = (k+1) * alpha_minus + np.log1p(-np.exp(-k * alpha_minus))
    T2_log_denom = alpha_minus + np.log1p(-np.exp(-alpha_minus))

    T2_log = T2_log_num - T2_log_denom + common_factor_log

    T_max_log = np.maximum(T1_log, T2_log)

    error_term = np.exp(T_max_log) * (np.exp(T1_log - T_max_log) + np.exp(T2_log - T_max_log))

    return error_term

def _delta_fft_computations(omegas: np.ndarray, target_eps: float, num_compositions: int, L: float):
    """ Core computation of privacy loss distribution convolutions using FFT. """
    # Flip omegas, i.e. fx <- D(omega_y), the matrix D = [0 I;I 0]
    nx = len(omegas)
    half = nx // 2
    fx = np.concatenate((omegas[half:], omegas[:half]))
    assert np.size(fx) == np.size(omegas)

    # Compute the DFT
    FF1 = np.fft.rfft(fx)

    # Take elementwise powers and compute the inverse DFT
    cfx = np.real(np.fft.irfft((FF1 ** num_compositions)))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    cfx = np.concatenate((cfx[half:], cfx[:half]))

    # assert np.allclose(np.sum(cfx), 1), "sum over convolved pld is not one!"

    # Evaluate \delta(target_eps)
    x = np.linspace(-L, L, nx, endpoint=False) # grid for the numerical integration
    exp_e = 1 - np.exp(target_eps - x)
    integrand = exp_e[exp_e > 0] * cfx[exp_e > 0]
    assert np.all(~(integrand < 0 ) | np.isclose(integrand, 0)), "encountered negative values in pld after composition"

    sum_int = np.sum(integrand)
    return sum_int

def get_delta_upper_bound(
        pld: PrivacyLossDistribution,
        target_eps: float,
        num_compositions: int,
        num_discretisation_points: int = int(1E6),
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
        - target_eps: The privacy parameters ε for which to compute δ.
        - num_compositions: The number of compositions (=applications) of the privacy mechanism.
        - num_discretisation_points: The number of discretisation points used by the accountant.
        - L: The truncation threshold (in privacy loss space) used by the accountant.
    """
    nx = int(num_discretisation_points)

    # Evaluate the bound of Thm. 10
    error_term = get_delta_error_term(pld, num_compositions, L, nx)

    _, omega_y, _ = pld.discretize_privacy_loss_distribution(-L, L, nx)

    delta = _delta_fft_computations(omega_y, target_eps, num_compositions, L)
    delta += error_term

    return delta

def get_delta_lower_bound(
        pld: PrivacyLossDistribution,
        target_eps: float,
        num_compositions: int,
        num_discretisation_points: int = int(1E6),
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
        - target_eps: The privacy parameters ε for which to compute δ.
        - num_compositions: The number of compositions (=applications) of the privacy mechanism.
        - num_discretisation_points: The number of discretisation points used by the accountant.
        - L: The truncation threshold (in privacy loss space) used by the accountant.
    """
    nx = int(num_discretisation_points)

    error_term = get_delta_error_term(pld, num_compositions, L, nx)

    omega_y, _, _ = pld.discretize_privacy_loss_distribution(-L, L, nx)

    delta = _delta_fft_computations(omega_y, target_eps, num_compositions, L)
    delta -= error_term

    return delta


if __name__ == '__main__':
    print(get_delta_upper_bound(ExponentialMechanismPrivacyLossDistribution(.1, 7, 10), target_eps=1., num_compositions=1000))
    print(get_delta_lower_bound(ExponentialMechanismPrivacyLossDistribution(.1, 7, 10), target_eps=1., num_compositions=1000))

    q = 0.01
    sigma = 2
    print(get_delta_upper_bound(SubsampledGaussianMechanismPrivacyLossDistribution(sigma, q), target_eps=1., num_compositions=1000, L=20))
    print(get_delta_lower_bound(SubsampledGaussianMechanismPrivacyLossDistribution(sigma, q), target_eps=1., num_compositions=1000, L=20))

