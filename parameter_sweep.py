import fourier_accountant
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--skip-q", action='store_true')
parser.add_argument("--skip-sigma", action='store_true')
parser.add_argument("--skip-ncomp", action='store_true')
parser.add_argument("--skip-eps", action='store_true')
parser.add_argument("--skip-bins", action='store_true')
parser.add_argument("--skip-lambdas", action='store_true')
parser.add_argument("--skip-L", action='store_true')
args = parser.parse_args()

# # sweep L for large q
# if not args.skip_L:
#     print("#### sweep L")
#     num_bins = int(4e7)
#     ncomp = 10000 # number of compositions of DP queries over minibatches = number of iterations of SGD
#     sigma = 4.0   # noise level for each query
#     target_eps = 1.0
#     q = 1.
#     Ls = [20, 40, 80]
#     print(f"{q=} {sigma=} {ncomp=} {target_eps=} {num_bins=}\n")
#     pld = fourier_accountant.plds.SubsampledGaussianMechanism(
#             sigma, q, fourier_accountant.plds.NeighborRelation.REMOVE_POISSON
#         )

#     for L in Ls:
#         try:
#             print(f"{L=}:", end="", flush=True)
#             delta_upper = fourier_accountant.get_delta_upper_bound(
#                 pld, target_eps, ncomp, L=L, num_discretisation_bins_half=num_bins
#             )
#             print(".", end="", flush=True)
#             delta_lower = fourier_accountant.get_delta_lower_bound(
#                 pld, target_eps, ncomp, L=L, num_discretisation_bins_half=num_bins
#             )
#             print(".", end="", flush=True)
#             print(f" {delta_lower=}, {delta_upper=}")
#         except Exception as e:
#             print(f" EXCEPTION ({type(e)}) {e}")
#         print("")
# exit(1)

# sweep lambdas in error
if not args.skip_lambdas:
    print("### sweep lambdas")
    num_bins = int(4e7)
    ncomp = 10000 # number of compositions of DP queries over minibatches = number of iterations of SGD
    sigma = 4.0   # noise level for each query
    target_eps = 1.0
    # q = 1.
    q = 0.001
    L = 20.
    print(f"{q=} {sigma=} {ncomp=} {num_bins=} {target_eps=}")
    lambdas = np.array([0.01, 0.1, .5, 1., 2., 10, 100]) * L

    pld = fourier_accountant.plds.SubsampledGaussianMechanism(
            sigma, q, fourier_accountant.plds.NeighborRelation.REMOVE_POISSON
        )

    omega_y_L, omega_y_R, Lxs = pld.discretize_privacy_loss_distribution(-L, L, num_bins)
    convolved_omegas = fourier_accountant.accountant._delta_fft_computations(omega_y_L, ncomp)
    ps, Lxs = fourier_accountant.accountant._get_ps_and_Lxs(pld, omega_y_R, Lxs)

    for lambd in lambdas:
        error_term = fourier_accountant.accountant._get_delta_error_term(Lxs, ps, ncomp, L, lambd)
        print(f"{lambd=}: {error_term=}\n")


# sweep number of discretisation bins
if not args.skip_bins:
    print("#### sweep bins")
    num_bins_base = int(1e7)
    ncomp = 10000 # number of compositions of DP queries over minibatches = number of iterations of SGD
    sigma = 4.0   # noise level for each query
    target_eps = 1.0
    factors = [1, 2, 4, 8]
    q = 0.01
    print(f"{q=} {sigma=} {ncomp=} {target_eps=}\n")
    pld = fourier_accountant.plds.SubsampledGaussianMechanism(
            sigma, q, fourier_accountant.plds.NeighborRelation.REMOVE_POISSON
        )

    for factor in factors:
        num_bins = num_bins_base * factor
        try:
            print(f"{num_bins=}:", end="", flush=True)
            delta_upper = fourier_accountant.get_delta_upper_bound(
                pld, target_eps, ncomp, L=20, num_discretisation_bins_half=num_bins
            )
            print(".", end="", flush=True)
            delta_lower = fourier_accountant.get_delta_lower_bound(
                pld, target_eps, ncomp, L=20, num_discretisation_bins_half=num_bins
            )
            print(".", end="", flush=True)
            print(f" {delta_lower=}, {delta_upper=}")
        except Exception as e:
            print(f" EXCEPTION ({type(e)}) {e}")
        print("")

if not args.skip_q:
    print("#### sweep q")
    ncomp = 10000 # number of compositions of DP queries over minibatches = number of iterations of SGD
    sigma = 4.0   # noise level for each query
    target_eps = 1.0
    print(f"{sigma=} {ncomp=} {target_eps=}\n")
    qs    = [0.001, 0.01, 0.1, 0.5, 1]  # subsampling ratio of minibatch

    for q in qs:
        pld = fourier_accountant.plds.SubsampledGaussianMechanism(
            sigma, q, fourier_accountant.plds.NeighborRelation.REMOVE_POISSON
        )

        try:
            print(f"{q=}:", end="", flush=True)
            delta_upper = fourier_accountant.get_delta_upper_bound(
                pld, target_eps, ncomp, L=20, num_discretisation_bins_half=int(1E7)
            )
            print(".", end="", flush=True)
            delta_lower = fourier_accountant.get_delta_lower_bound(
                pld, target_eps, ncomp, L=20, num_discretisation_bins_half=int(1E7)
            )
            print(".", end="", flush=True)
            delta_old = fourier_accountant.get_delta_R(
                target_eps, sigma, q, ncomp, nx=int(2E7), L=20
            )
            print(".", end="", flush=True)
            print(f" {delta_lower=}, {delta_upper=} ({delta_old=})")
        except Exception as e:
            print(f" EXCEPTION ({type(e)}) {e}")
        print("")


# sweep sigma
if not args.skip_sigma:
    print("#### sweep sigma")
    ncomp = 10000 # number of compositions of DP queries over minibatches = number of iterations of SGD
    q = 0.01
    target_eps = 1.0
    print(f"{q=} {ncomp=} {target_eps=}\n")
    sigmas = [0, 0.01, 0.5, 1, 4, 10, 100]

    for sigma in sigmas:
        pld = fourier_accountant.plds.SubsampledGaussianMechanism(
            sigma, q, fourier_accountant.plds.NeighborRelation.REMOVE_POISSON
        )

        try:
            print(f"{sigma=}:", end="", flush=True)
            delta_upper = fourier_accountant.get_delta_upper_bound(
                pld, target_eps, ncomp, L=20, num_discretisation_bins_half=int(1E7)
            )
            print(".", end="", flush=True)
            delta_lower = fourier_accountant.get_delta_lower_bound(
                pld, target_eps, ncomp, L=20, num_discretisation_bins_half=int(1E7)
            )
            print(".", end="", flush=True)
            delta_old = fourier_accountant.get_delta_R(
                target_eps, sigma, q, ncomp, nx=int(2E7), L=20
            )
            print(".", end="", flush=True)
            print(f" {delta_lower=}, {delta_upper=} ({delta_old=})")
        except Exception as e:
            print(f" EXCEPTION ({type(e)}) {e}")
        print("")

# sweep ncomp
if not args.skip_ncomp:
    print("#### sweep ncomp")
    q = 0.01
    sigma = 4.0
    target_eps = 1.0
    print(f"{q=} {sigma=} {target_eps=}\n")
    ncomps = [1, 100, 10000, 1000000, 100000000]

    pld = fourier_accountant.plds.SubsampledGaussianMechanism(
        sigma, q, fourier_accountant.plds.NeighborRelation.REMOVE_POISSON
    )

    for ncomp in ncomps:

        try:
            print(f"{ncomp=}:", end="", flush=True)
            delta_upper = fourier_accountant.get_delta_upper_bound(
                pld, target_eps, ncomp, L=20, num_discretisation_bins_half=int(1E7)
            )
            print(".", end="", flush=True)
            delta_lower = fourier_accountant.get_delta_lower_bound(
                pld, target_eps, ncomp, L=20, num_discretisation_bins_half=int(1E7)
            )
            print(".", end="", flush=True)
            delta_old = fourier_accountant.get_delta_R(
                target_eps, sigma, q, ncomp, nx=int(2E7), L=20
            )
            print(".", end="", flush=True)
            print(f" {delta_lower=}, {delta_upper=} ({delta_old=})")
        except Exception as e:
            print(f" EXCEPTION ({type(e)}) {e}")
        print("")



# sweep eps
if not args.skip_eps:
    print("#### sweep eps")
    q = 0.01
    sigma = 4.0
    ncomp = 10000
    print(f"{q=} {sigma=} {ncomp=}\n")

    target_epss = [0.01, 0.1, 1.0, 4.0, 10.0, 100.0]

    pld = fourier_accountant.plds.SubsampledGaussianMechanism(
        sigma, q, fourier_accountant.plds.NeighborRelation.REMOVE_POISSON
    )

    for target_eps in target_epss:

        try:
            print(f"{target_eps=}:", end="", flush=True)
            delta_upper = fourier_accountant.get_delta_upper_bound(
                pld, target_eps, ncomp, L=20, num_discretisation_bins_half=int(1E7)
            )
            print(".", end="", flush=True)
            delta_lower = fourier_accountant.get_delta_lower_bound(
                pld, target_eps, ncomp, L=20, num_discretisation_bins_half=int(1E7)
            )
            print(".", end="", flush=True)
            delta_old = fourier_accountant.get_delta_R(
                target_eps, sigma, q, ncomp, nx=int(1E7), L=20
            )
            print(".", end="", flush=True)
            print(f" {delta_lower=}, {delta_upper=} ({delta_old=})")
        except Exception as e:
            print(f" EXCEPTION ({type(e)}) {e}")
        print("")
