"""GPHMC 2D example"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from gphmc.gaussian_process_regression.gaussian_process.covariance import (
    SquaredExponential
)
from gphmc.gaussian_process_regression.gaussian_process.gp import (
    GaussianProcess
)
from gphmc.gaussian_process_regression.gaussian_process.optimizer import (
    SECovLikelihoodOptimizer
)
from gphmc.gaussian_process_regression.gaussian_process.util import (
    get_logger
)
from gphmc.gphmc import GPHMCSampler

FUZZ = 1e-300


def potential_energy(x):
    """Density to estimate

    :param x: input parameter
    :return: value of the density at (x, y)
    """

    probability = np.exp(-8*(x[0]**2/2+x[1]**2-1)**2) + FUZZ
    energy = -np.log(probability)

    return energy


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
        list_y.append(potential_energy((i, j)))

    return list_observations, list_y


def plot_density(n=50):
    """Plot the density

    :param n: number of point on each axis to estimate the density on.
     The density will be estimated on a regular n x n grid n the [-2, 2, -2, 2]
     space.
    """

    list_x = np.linspace(-2, 2, n)
    coords = np.meshgrid(list_x, list_x, indexing='ij')
    density_value = np.exp(-potential_energy([coords[0], coords[1]]))
    plt.imshow(
        density_value, cmap='viridis',
        extent=[-2, 2, -2, 2], vmin=0.0, vmax=1.0, aspect='equal'
    )
    plt.title('Real Density')
    plt.savefig('figures/density.png', dpi=100, bbox_inches='tight')
    plt.close()


def plot_gp(current_cov_param, list_obs, list_y, n=50):
    """Plot the Gaussian process estimation of the density

    :param current_cov_param: tuple with the 3 parameters of the exponential
     covariance matrix
    :param list_obs: list of 2-tuple corresponding to the point observed
    :param list_y: list of scalar corresponding to the density value at the
     observed points
    :param n: number of point on each axis to estimate the density on.
     The density will be estimated on a regular n x n grid n the [-2, 2, -2, 2]
     space.
    """

    cov = SquaredExponential(current_cov_param[0], current_cov_param[1:])
    gp = GaussianProcess(cov, list_observations=list_obs,
                         list_y=list_y, noise=1e-3)
    gp.covariance_matrix()
    list_x = np.linspace(-2, 2, n)
    current_estimation = np.zeros((n, n))
    current_estimation_flat = [
        (i, j, gp.mean([(xi, yj)])[0])
        for (i, xi) in enumerate(list_x)
        for (j, yj) in enumerate(list_x)
    ]
    for coord_value in current_estimation_flat:
        current_estimation[coord_value[:2]] = np.exp(-coord_value[2])
    plt.imshow(current_estimation, cmap='viridis', vmin=0, vmax=1,
               extent=[-2, 2, -2, 2], aspect='equal')
    idx_x = np.array([abs(x[0]) <= 1.8 for x in gp.list_observations])
    idx_y = np.array([abs(x[1]) <= 1.8 for x in gp.list_observations])
    idx = idx_x & idx_y
    x = np.array([-x[0] for x in gp.list_observations])
    y = np.array([x[1] for x in gp.list_observations])
    z = np.array([np.exp(-y) for y in gp.list_y])
    plt.scatter(
        y[idx], x[idx], c=z[idx], cmap='viridis', s=100, vmin=0, vmax=1
    )
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title('Gaussian process after exploration')
    plt.savefig('figures/estimation.png', dpi=100, bbox_inches='tight')
    plt.close()


def plot_samples(samples):
    """Plot the final samples

    :param samples: list of tuple corresponding to the samples of the GPHMC
    """

    x = [-sample[0] for sample in samples]
    y = [sample[1] for sample in samples]
    plt.imshow(np.ones((50, 50)), cmap='gray', vmin=0, vmax=1,
               extent=[-2, 2, -2, 2], aspect='equal')
    plt.scatter(y, x, s=100, c='w')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title('Final samples')
    plt.savefig('figures/sample.png', dpi=100, bbox_inches='tight')
    plt.close()


def main():
    """Main"""

    np.random.seed(0)

    plot_density()
    logger = get_logger()
    logger.info('Get initial observations')
    list_obs, list_y = get_observations(xmin=-2, xmax=2, ymin=-2, ymax=2, n=50)
    logger.info('Instantiate the sampler')
    sampler = GPHMCSampler(
        covariance_class=SquaredExponential, target_function=potential_energy,
        likelihood_optimizer_class=SECovLikelihoodOptimizer,
        list_obs=list_obs, list_y=list_y, noise=1e-3, dimension=2, n_explo=50,
        init_cov_param=np.array([1, 1, 1])
    )
    logger.info('Exploration phase')
    sampler.exploration(
        epsilon=0.1, length=400, momentum_std=1.0, gp_update_rate=10
    )
    plot_gp(sampler.current_cov_param, sampler.list_obs, sampler.list_y)
    logger.info('Sampling phase')
    samples = []
    sample_generator = sampler.sample(
        epsilon=0.01, length=200, momentum_std=1.0
    )
    for _ in tqdm(range(500)):
        samples.append(next(sample_generator))
    plot_samples(samples)


if __name__ == '__main__':

    main()
