import numpy as np
from pld import *

class RandomizedResponsePrivacyLossDistribution(DiscretePrivacyLossDistribution):
    """
    PLD for randomized response with privacy value eps_rr
    """

    def __init__(self, eps_em: float, p: float) -> None:
        """
        Args:
            eps_rr: The epsilon value of the mechanism under composition.
            p: probability of flipping the bit.
        """
        p1 = np.array([(1.-p), p])
        p2 = np.array([p, (1.-p)])

        super().__init__(p1, p2)

    def get_accountant_parameters(self, error_tolerance: float) -> typing.Any:
        super().get_accountant_parameters(error_tolerance)

p = 0.7
eps = .1
rr_pld = RandomizedResponsePrivacyLossDistribution(eps, p)
delta = get_delta_upper_bound(rr_pld, target_eps = eps, num_compositions = 1)
delta_error = get_delta_upper_bound(rr_pld, num_compositions = 1)
cp = np.log(p) - np.log1p(-p)
analytical_delta = p * (1. - np.exp(eps - cp))
