# PLD-Accountant

Python code for computing exact DP-guarantees for the subsampled Gaussian mechanism.  

The method is described in:

Antti Koskela, Joonas Jälkö, Antti Honkela:  
Computing Exact Guarantees for Differential Privacy  
https://arxiv.org/abs/1906.03049  

# Usage

Examples of usage are given in the file test_eps_delta.py.


You can download a Python package by running

pip3 install pld-accountant

Then, run for example:


from pld_accountant import compute_eps,compute_delta

q=0.01
sigma=1.2
nc=1000 #number of compositions

delta=1e-5

a  = compute_eps.get_epsilon_bounded(q=q,sigma=sigma,target_delta=delta,ncomp=nc)

eps=2.0

d  = compute_delta.get_delta_bounded(q=q,sigma=sigma,target_eps=eps,ncomp=nc)
