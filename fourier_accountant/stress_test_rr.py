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

## single iteration
p = 0.7
eps = .1
T = 1
rr_pld = RandomizedResponsePrivacyLossDistribution(eps, p)
delta_upper = get_delta_upper_bound(rr_pld, target_eps = eps, num_compositions = T)
delta_lower = get_delta_lower_bound(rr_pld, target_eps = eps, num_compositions = T)
delta_error = get_delta_error_term(rr_pld, num_compositions = T)
cp = np.log(p) - np.log1p(-p)
analytical_delta = p * (1. - np.exp(eps - cp))
assert(analytical_delta < delta_upper)
# it seems possible that the lower is also larger than the analytical value
assert(analytical_delta < delta_lower)
##

## more iterations
"""
With randomized response, if eps -> cp, the delta goes to 0.
Now if we naively compose these mechanisms T times, we should get (T*cp,0)-DP
Lets test how PLD operates close to pure eps mechanisms. We set T*cp as the 
target eps, and hope to recover deltas close to 0.
"""

# this works
p = 0.8
T = 10
cp = np.log(p) - np.log1p(-p)
eps = cp * T
rr_pld = RandomizedResponsePrivacyLossDistribution(eps, p)
delta_upper = get_delta_upper_bound(rr_pld, target_eps = eps, num_compositions = T)
delta_error = get_delta_error_term(rr_pld, num_compositions = T)
print(delta_upper, delta_error)
# this doesn't
p = 0.9
T = 10
cp = np.log(p) - np.log1p(-p)
eps = cp * T
rr_pld = RandomizedResponsePrivacyLossDistribution(eps, p)
delta_upper = get_delta_upper_bound(rr_pld, target_eps = eps, num_compositions = T)
delta_error = get_delta_error_term(rr_pld, num_compositions = T)
print(delta_upper, delta_error)
# there seems to be rather steep increase in delta when T increases from 8 to 9
for T in range(11):
	p = 0.9
	cp = np.log(p) - np.log1p(-p)
	eps = cp * T
	rr_pld = RandomizedResponsePrivacyLossDistribution(eps, p)
	delta_upper = get_delta_upper_bound(rr_pld, target_eps = eps, num_compositions = T)
	delta_lower = get_delta_lower_bound(rr_pld, target_eps = eps, num_compositions = T)
	delta_error = get_delta_error_term(rr_pld, num_compositions = T)
	print("With {}: delta_lower={}, delta_upper={}".format(T, delta_lower, delta_upper))
##
