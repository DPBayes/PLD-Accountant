from abc import ABCMeta, abstractmethod, abstractproperty
import typing
import numpy as np
import scipy.special

class PrivacyLossDistribution(metaclass=ABCMeta):

    @abstractmethod
    def get_accountant_parameters(self, error_tolerance: float) -> typing.Any:
        """ Determines suitable hyperparameters for the Fourier accountant for a given error tolerance. """

    @abstractproperty
    def privacy_loss_values(self) -> np.ndarray:
        """ The values of the privacy loss random variable over the DP mechanisms output domain.

        Not ordered and not guaranteed to be free of duplicates.
        """

    @abstractproperty
    def privacy_loss_probabilities(self) -> np.ndarray:
        """ The probability mass omega associated with each privacy loss value. """

    @abstractmethod
    def discretize_privacy_loss_distribution(self, start: float, stop: float, step: float) -> np.ndarray:
        """ Returns the associated probability masses associated with each value of a given discretised interval in privacy loss space. """

class DiscretePrivacyLossDistribution(PrivacyLossDistribution):
    """ The privacy loss distribution defined by two discrete probability mass functions. """

    def __init__(self, p1, p2) -> None:
        """ indices in p1/p2 must correspond. ( what exactly do p1/p2 give probabilities for? ) """
        if np.size(p1) != np.size(p2):
            raise ValueError("Both probability mass distributions must have the same size.")

        self._p1 = np.array(p1)
        self._p2 = np.array(p2)

    def get_accountant_parameters(self, error_tolerance: float) -> typing.Any:
        raise NotImplementedError()

    @property
    def privacy_loss_values(self) -> np.ndarray:
        return np.log(self._p1 / self._p2)

    @property
    def privacy_loss_probabilities(self) -> np.ndarray:
        return self._p1

    def discretize_privacy_loss_distribution(self, start: float, stop: float, number_of_discretisation_points: int) -> np.ndarray:
        nx = number_of_discretisation_points
        dx = (stop - start) / nx

        Lx = self.privacy_loss_values
        ps = self.privacy_loss_probabilities

        omega_y = np.zeros(nx)
        for lx, p in zip(Lx, ps): # todo(lumip): can optimise?
            ii = int(np.ceil((lx - start) / dx))
            omega_y[ii] += p

        return omega_y

class ExponentialMechanismPrivacyLossDistribution(DiscretePrivacyLossDistribution):
    """
    PLD for exponential mechanism with privacy value eps_em
    where the quality score is a counting query.
    """

    def __init__(self, eps_em: float, m: int, n: int) -> None:
        """
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

def get_delta_error_term(
        pld: PrivacyLossDistribution,
        num_compositions: int = 500,
        L: float = 20.0,
        lambd: typing.Optional[float] = None
    ):

    # Determine the privacy loss function
    Lx = pld.privacy_loss_values
    ps = pld.privacy_loss_probabilities
    assert np.size(ps) == np.size(Lx)

    if lambd is None:
        lambd = .5 * L

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
    # T1 = (2 * np.exp((k + 1) * alpha_plus) - np.exp(k * alpha_plus) - np.exp(alpha_plus) ) / (np.exp(alpha_plus) - 1)

    T2_log_num = (k+1) * alpha_minus + np.log1p(-np.exp(-k * alpha_minus))
    T2_log_denom = alpha_minus + np.log1p(-np.exp(-alpha_minus))

    T2_log = T2_log_num - T2_log_denom + common_factor_log
    # T2 = (np.exp((k + 1) * alpha_minus) - np.exp(alpha_minus) ) / (np.exp(alpha_minus) - 1)
    # error_term = (T1 + T2) * (np.exp(-lambd*L)/(1-np.exp(-lambd*L)))

    T_max_log = np.maximum(T1_log, T2_log)

    error_term = np.exp(T_max_log) * (np.exp(T1_log - T_max_log) + np.exp(T2_log - T_max_log))

    return error_term

def get_delta_upper_bound(
        pld: PrivacyLossDistribution,
        target_eps: float = 1.0,
        num_compositions: int = 500,
        num_discretisation_points: int = 1E6,
        L: float = 20.0
    ):
    """
    Calculates the upper bound for delta given a target epsilon for
    k-fold composition of any discrete mechanism.

    Args:
        target_eps: The targeted value for epsilon of the composition.
        num_compositions: Number of compositions of the mechanism.
        num_discretisation_points: Number of points in the discretisation grid.
        L: Limit for the approximation integral.
    """

    # Evaluate the bound of Thm. 10
    error_term = get_delta_error_term(pld, num_compositions, L)

    nx = int(num_discretisation_points)

    omega_y = pld.discretize_privacy_loss_distribution(-L, L, nx)

    # Flip omega_y, i.e. fx <- D(omega_y), the matrix D = [0 I;I 0]
    half = nx // 2
    fx = np.concatenate((omega_y[half:], omega_y[:half]))
    assert np.size(fx) == np.size(omega_y)

    # Compute the DFT
    FF1 = np.fft.fft(fx)

    # Take elementwise powers and compute the inverse DFT
    cfx = np.real(np.fft.ifft((FF1 ** num_compositions)))

    # Flip again, i.e. cfx <- D(cfx), D = [0 I;I 0]
    cfx = np.concatenate((cfx[half:], cfx[:half]))

    assert np.allclose(np.sum(cfx), 1), "sum over convolved pld is not one!"

    # Evaluate \delta(target_eps)
    x = np.linspace(-L, L, nx, endpoint=False) # grid for the numerical integration
    exp_e = 1 - np.exp(target_eps - x)
    integrand = exp_e * cfx
    sum_int = np.sum(integrand[exp_e > 0])
    delta = sum_int + error_term

    return delta


if __name__ == '__main__':
    get_delta_upper_bound(ExponentialMechanismPrivacyLossDistribution(.1, 7, 10))
