
from fourier_accountant.experimental.binomial_mechanism import get_epsilon

f_diff = 0.1 # \Delta = f(X) - f(Y) from Thm 11. of https://arxiv.org/abs/2006.07134,
             # where f is the function operating on data sets X and Y that is masked
             # via the binomial mechanism
n = 100      # binomial mechanism parameter n
p = 0.5      # binomial mechanism parameter p
s = 0.5      # binomial mechanism parameter s

get_epsilon(n, p, s, f_diff)