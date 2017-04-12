"""GPHMC"""
import numpy as np

from gaussian_process_regression.gaussian_process.covariance import Covariance
from gaussian_process_regression.gaussian_process.optimizer import (
    LikelihoodOptimizer
)


class NoExplorationError(Exception):
    """Raised when the exploration stage has been skipped"""
    pass


class GPHMCSampler:

    def __init__(self, dimension, covariance_class: Covariance, n_explo,
                 epsilon, dynamics_length,
                 likelihood_optimizer_class: LikelihoodOptimizer, list_obs,
                 list_y, init_cov_param, noise, target_function):
        """Init

        :param dimension: int, the dimension of the parameter space
        :param covariance_class: the class of the covariance matrix
        :param n_explo: the number of exploratory space to refine the Gaussian
         process
        :param epsilon: the numerical parameter of the leapfrog scheme
        :param dynamics_length: the length of the Hamiltonian dynamics
        :param likelihood_optimizer_class: the class of the likelihood
         optimizer
        :param list_obs: the initial list of observation
        :param list_y: the initial list of target function evaluation
        :param init_cov_param: the initial parameters of the covariance
         function
        :param noise: the noise
        :param target_function: the target function to be sampled (i.e. the
         negative log lokelihood of the density probability of interest)
        """

        self.likelihood_optimizer = likelihood_optimizer_class(
            covariance_class, list_obs, list_y, initial_guess=init_cov_param,
            noise=noise
        )
        self.dimension = dimension
        self.n_explo = n_explo
        self.epsilon = epsilon
        self.dynamics_length = dynamics_length
        self.list_obs = list_obs
        self.list_y = list_y
        self.current_cov_param = init_cov_param
        self.target_function = target_function
        self.init_y = None
        self.init_obs = None
        self.gp = None

        self._assert_dimension()

    def _assert_dimension(self):

        for obs, y in zip(self.list_obs, self.list_y):
            assert len(y) == self.dimension
            assert len(obs) == self.dimension

    def _add_observation(self, obs_end):

        self.list_obs.append(tuple(obs_end))
        y_end = self.target_function(obs_end)
        self.list_y.append(y_end)

        return y_end

    def _leapfrog_dynamics(self, type='sampling'):

        obs = self.init_obs.copy()
        momentum_start = np.random.randn(self.dimension)
        momentum = momentum_start.copy()
        momentum -= self.gp_grad(obs, type=type)*self.epsilon/2
        for idx in range(self.dynamics_length):
            obs += momentum*self.epsilon
            if idx < self.dynamics_length-1:
                momentum -= self.gp_grad(obs, type=type)*self.epsilon
        momentum -= self.gp_grad(obs, type=type)*self.epsilon/2

        result = {
            'obs_1': self.init_obs,
            'obs_2': obs,
            'momentum_1': momentum_start,
            'momentum_2': momentum
        }

        return result

    def gp_grad(self, obs, type):

        gp_grad = np.array([
            self.gp.mean([tuple(obs)], derivative=True, i=i)
            for i in range(self.dimension)
        ])

        if type == 'exploration':
            sigma_grad = np.array([
                self.gp.sigma([tuple(obs)], derivative=True, i=i)
                for i in range(self.dimension)
            ])
            gp_grad -= sigma_grad

        return gp_grad

    def _fit_covariance_parameters(self):

        res = self.likelihood_optimizer.maximum_likelihood()
        self.current_cov_param = res.x

    def _update_position(self, leapfrog_output, y_1, y_2):

        kin_energy_1 = np.sum(
            np.power(leapfrog_output['momentum_1'], 2)
        )/2
        kin_energy_2 = np.sum(
            np.power(leapfrog_output['momentum_2'], 2)
        ) / 2

        if np.random.rand(1) < np.exp(y_1-y_2+kin_energy_1-kin_energy_2):
            return leapfrog_output['obs_2'], y_2
        else:
            return leapfrog_output['obs_1'], y_1

    def _update_current_sample(self, type):

        leapfrog_output = self._leapfrog_dynamics(type=type)
        y_end = self._add_observation(leapfrog_output['obs_2'])
        self.init_obs, self.init_y = self._update_position(
            leapfrog_output, self.init_y, y_end
        )

    def exploration(self):

        self._fit_covariance_parameters()

        self.init_y = np.min[self.list_y]
        self.init_obs = self.list_obs[np.argmin[self.list_y]]
        for _ in range(self.n_explo):
            self._update_current_sample('exploration')
            self._fit_covariance_parameters()

        self.gp = self.likelihood_optimizer.instanciate_gp(
            self.current_cov_param
        )

    def sample(self):

        if not (self.init_obs and self.init_y and self.gp):
            raise NoExplorationError

        self._update_current_sample('sampling')

        yield self.init_obs
