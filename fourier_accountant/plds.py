from abc import ABCMeta, abstractmethod, abstractproperty
import typing
import numpy as np
import scipy.special
import scipy.optimize
from enum import Enum
import warnings


class NeighborRelation(Enum):
    REMOVE_POISSON = 'remove-poisson'
    SUBSTITUTE_NO_REPLACE = 'substitute-no-replace'

class PrivacyException(Exception):
    """ An exception indicating a violation of privacy constraints. """

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

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
    """

    def __init__(self,
            sigma: float, q: typing.Optional[float] = 1., relation: NeighborRelation = NeighborRelation.REMOVE_POISSON
        ) -> None:
        """
        Args:
            - sigma: Gaussian mechanism noise level (without factoring in sensitivity or subsampling).
            - q: Subsampling ratio.
            - relation: The neighboring relation for datasets.
        """
        self.sigma = np.abs(sigma)
        self.q = q
        self._evaluate_internals = None
        if self.q <= 0 or self.q > 1:
            raise ValueError(f"Subsampling ratio q must be larger than 0 and less than or equal to 1, was {q}.")
        if relation == NeighborRelation.REMOVE_POISSON:
            self._evaluate_internals = self._evaluate_internals_remove_relation
        elif relation == NeighborRelation.SUBSTITUTE_NO_REPLACE:
            self._evaluate_internals = self._evaluate_internals_substitute_relation
        else:
            raise ValueError("Unknown neighboring relation given.")


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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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

    def _evaluate_internals_remove_relation(self,
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

        mask = np.ones_like(x, dtype=bool)
        if q < 1:
            mask = x > np.log(1 - q)
            # note(lumip): actually we'd be fine with q=1 computing log(1-q)=-inf
            #   giving us the full mask, but numpy will spam warnings, which
            #   would confuse users, therefore this construct is necessary

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

    def _evaluate_internals_substitute_relation(self,
            x: typing.Sequence[float], compute_derivative: typing.Optional[bool]=False
        ) -> typing.Union[np.array, typing.Tuple[np.array, np.array]]:
        sigma = self.sigma
        q = self.q

        mask = np.ones_like(x, dtype=bool)
        if q < 1:
            mask = x > np.log(1 - q)

        sigma_sq = sigma**2
        exp_x = np.exp(x[mask])

        c = q * np.exp( -1 / (2 * sigma_sq) )
        c_sq_exp_x_4 = 4 * c**2 * exp_x

        sqrtpart = np.sqrt( ((1 - q) * (1 - exp_x))**2 + c_sq_exp_x_4 )
        logpart = ( sqrtpart - (1 - q) * (1 - exp_x) ) / (2 * c)
        assert np.all(logpart > 0.)
        Linvx = sigma_sq * np.log(logpart) # L^{-1}(s)

        # note: straightforward implementation of derivative dLinvx:
        # dlogpart_left = ((1 - q) * exp_x) / (2 * c)
        # dlogpart_right = c_sq_exp_x_4 - 2 * (1 - q)**2 * exp_x * (1 - exp_x)
        # dlogpart_right /= (4 * c * sqrtpart)
        # dLinvx = (sigma_sq / logpart) * (dlogpart_left + dlogpart_right)

        # note: slightly massaged implementation of derivative dLinvx:
        dsqrtpart = c_sq_exp_x_4 - 2 * (1 - q)**2 * exp_x * (1 - exp_x)
        dsqrtpart /= 2*sqrtpart
        dlogpart = (1 - q) * exp_x + dsqrtpart # without factor 2*c
        # note: outer derivative of log would now require division by logpart,
        #       but for stability we multiply numerator and denominator by
        #       sqrtpart + (1-q) * (1-exp_x) to get the below
        dlogpart_multiplied = dlogpart * (sqrtpart + (1 - q) * (1 - exp_x))
        dLinvx = sigma_sq * dlogpart_multiplied / c_sq_exp_x_4 # d/ds L^{-1}(s)

        # f_X(L^{-1}(s)):
        ALinvx = (1 / np.sqrt(2 * np.pi * sigma**2) ) * (
                    (1 - q) * np.exp(-Linvx**2     / (2 * sigma_sq)) +
                    q *       np.exp(-(Linvx-1)**2 / (2 * sigma_sq))
            )

        omega = np.zeros_like(x)
        omega[mask] = ALinvx * dLinvx

        if not compute_derivative:
            return omega

        raise NotImplementedError("Derivative for substitute relation currently not implemented.")

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
