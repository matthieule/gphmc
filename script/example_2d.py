"""GPHMC 2D example"""
import numpy as np

from gphmc.gphmc import GPHMCSampler
from gaussian_process_regression.gaussian_process.covariance import (
    SquaredExponential
)
from gaussian_process_regression.gaussian_process.optimizer import (
    SECovLikelihoodOptimizer
)


def potential_energy(x):
    """Density to estimate

    :param x: input parameter
    :return: value of the density at (x, y)
    """

    probability = np.exp(-(1-(x[0]**2/2+x[1]**2/0.5))**2/0.5)
    potential_energy = -np.log(probability)
    return potential_energy


def get_observations(xmin, xmax, ymin, ymax, n):
    """Get a list of random observation

    :param xmin: minimum x value where to plot the density
    :param xmax: maximum x value where to plot the density
    :param ymin: minimum y value where to plot the density
    :param ymax: maximum y value where to plot the density
    :param n: number of observation
    :return: two lists of the same length. One list of the observation points,
     and one list of the evaluation of the density at the observation points
    """

    list_observations = []
    list_y = []
    for _ in range(n):
        i = np.random.uniform(low=xmin, high=xmax)
        j = np.random.uniform(low=ymin, high=ymax)
        list_observations.append((i, j))
        list_y.append(potential_energy(i, j))

    return list_observations, list_y


def main():

    list_obs, list_y = get_observations(
        xmin=-3, xmax=3, ymin=-3, ymax=3, n=15
    )
    GPHMCSampler(dimension=2, covariance_class=SquaredExponential, n_explo=100,
                 epsilon=0.05, dynamics_length=20,
                 likelihood_optimizer_class=SECovLikelihoodOptimizer,
                 list_obs=list_obs, list_y=list_y,
                 init_cov_param=np.array([1, 1, 1]),
                 noise=1e-6, target_function=potential_energy)


if __name__ == '__name__':

    main()
