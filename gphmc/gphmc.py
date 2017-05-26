"""GPHMC"""
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from .gaussian_process_regression.gaussian_process.covariance import Covariance
from .gaussian_process_regression.gaussian_process.optimizer import (
    LikelihoodOptimizer
)


class NoExplorationError(Exception):
    """Raised when the exploration stage has been skipped"""
    pass


class GPHMCSampler:

    def __init__(self, dimension, covariance_class: Covariance, n_explo,
                 likelihood_optimizer_class: LikelihoodOptimizer, list_obs,
                 list_y, init_cov_param, noise, target_function):
        """Init

        :param dimension: int, the dimension of the parameter space
        :param covariance_class: the class of the covariance matrix
        :param n_explo: the number of exploratory space to refine the Gaussian
         process
        :param likelihood_optimizer_class: the class of the likelihood
         optimizer
        :param list_obs: the initial list of observation
        :param list_y: the initial list of target function evaluation
        :param init_cov_param: the initial parameters of the covariance
         function
        :param noise: the noise
        :param target_function: the target function to be sampled (i.e. the
         negative log likelihood of the density probability of interest)
        """

        self.dimension = dimension
        self.n_explo = n_explo
        self._list_obs = deepcopy(list_obs)
        self._list_y = deepcopy(list_y)
        self.current_cov_param = init_cov_param.copy()
        self.target_function = target_function
        self.init_y = None
        self.init_obs = None
        self.gp = None

        # Note that we make use of the fact that the lists are passed by
        # reference here. We instantiate the likelihood_optimizer_class
        # with _list_y and _list_obs which will be updated as observations are
        # added
        self.likelihood_optimizer = likelihood_optimizer_class(
            covariance_class, self._list_obs, self._list_y,
            initial_guess=init_cov_param, noise=noise
        )

        self._assert_dimension()

    def _assert_dimension(self):
        """Assert that the input list have the correct length"""

        assert len(self._list_y) == len(self._list_obs)
        for obs, y in zip(self._list_obs, self._list_y):
            assert len(obs) == self.dimension

    def _fit_covariance_parameters(self, maxiter=1000, disp=True):
        """Fit the parameters of the covariance function"""

        res = self.likelihood_optimizer.maximum_likelihood(
            maxiter=maxiter, disp=disp
        )
        self.current_cov_param = res.x

    def _potential_energy_grad(self, obs, type):
        """Compute the gradient of the potential energy

        The potential energy is approximated with a Gaussian process. In the
        sampling phase, it is the mean of the Gaussian process. In the
        exploration phase, it is the mean minus the standard deviation.

        :param obs: array of dimension self.dimension
        :param type: 'sampling' or 'exploration'
        :return: the gradient of the potential energy and the standard
         deviation of the Gaussian process at obs
        """

        assert len(obs) == self.dimension
        assert type in ['sampling', 'exploration']

        potential_energy_grad = np.squeeze(np.array([
            self.gp.mean([tuple(obs)], derivative=True, i=i)
            for i in range(self.dimension)
        ]))
        sigma = np.squeeze(self.gp.sigma([tuple(obs)]))
        std = np.sqrt(sigma)

        # During the exploratory phase, the potential energy is the sum of the
        # normal potential energy with the opposite of the standard deviation
        # to explore space of high uncertainty
        if type == 'exploration':
            std_grad = np.squeeze(np.array([
                self.gp.sigma([tuple(obs)], derivative=True, i=i)
                for i in range(self.dimension)
            ]))
            std_grad /= 2*std
            potential_energy_grad -= std_grad

        return potential_energy_grad, std

    def _hamiltonian_dynamics(self, type='sampling', epsilon=0.1, length=100,
                              std_thr=3, momentum_std=1.0):
        """Hamiltonian dynamics using the leapfrog numerical scheme

        The momentum is drawn from a normal distribution

        :param type: 'sampling' or 'exploration'
        :param epsilon: time step of the Hamiltonian dynamics
        :param length: length of the Hamiltonian dynamics
        :param std_thr: threshold on the standard deviation when to stop the
         dynamics
        :param momentum_std: standard deviation of the Gaussian distribution
         the momentum is drawn from
        :return: dictionary with keys 'dynamics_end_obs' the point at the end
         of the dynamics, 'momentum_1' the momentum at the beginning of the
         dynamics, 'momentum_2' the momentum at the end of the dynamics, and
         'momentum_std' the standard deviation of the Gaussian distribution the
         momentum is drawn from
        """

        assert type in ['sampling', 'exploration']

        obs = deepcopy(self.init_obs)
        momentum_start = np.random.randn(self.dimension)*momentum_std
        momentum = momentum_start.copy()

        # Half step for the momentum
        pot_energy_grad, std = self._potential_energy_grad(obs, type=type)
        momentum -= pot_energy_grad*epsilon/2

        # Run the Hamiltonian dynamics
        for idx in range(length):
            obs += momentum*epsilon/momentum_std**2
            # For the last step, skip the momentum full update
            if idx < length-1:
                pot_energy_grad, std = (
                    self._potential_energy_grad(obs, type=type)
                )
                momentum -= pot_energy_grad*epsilon
            # If the standard deviation is greater than std_thr, stop the
            # dynamics
            if type == 'exploration' and std > std_thr:
                break
            if type == 'sampling':
                energy_pot = self.gp.mean([tuple(obs)])

        # Half step for the momentum
        pot_energy_grad, std = self._potential_energy_grad(obs, type=type)
        momentum -= pot_energy_grad*epsilon/2
        result = {
            'dynamics_end_obs': tuple(obs),
            'momentum_1': momentum_start,
            'momentum_2': momentum,
            'momentum_std': momentum_std
        }

        return result

    def _update_current_sample(self, type, epsilon=0.1, length=100,
                               momentum_std=1.0):
        """Update the current state of the sampler with Hamiltonian dynamics

        Run the Hamiltonian dynamics with the proposed parameters and
        update the init_obs and init_y of the sampler with a
        Metropolis-Hastings acceptance criteria

        :param type: 'sampling' or 'exploration'
        :param epsilon: time step of the Hamiltonian dynamics
        :param length: length of the Hamiltonian dynamics
        :param momentum_std: standard deviation of the Gaussian distribution
         the momentum is drawn from
        """

        assert type in ['sampling', 'exploration']

        dynamics_output = self._hamiltonian_dynamics(
            type=type, epsilon=epsilon, length=length,
            momentum_std=momentum_std
        )
        y_end = self.add_observation(dynamics_output['dynamics_end_obs'])

        self.init_obs, self.init_y = (
            self._metropolis_hastings_acceptance_criteria(
                dynamics_output, y_end
            )
        )

    def _metropolis_hastings_acceptance_criteria(self, dynamics_output, y_2):
        """Metropolis-Hastings acceptance criteria

        :param dynamics_output: Output of the hamiltonian dynamics: Dictionary
         with keys 'dynamics_end_obs' the point at the end of the dynamics,
         'momentum_1' the momentum at the beginning of the dynamics,
         'momentum_2' the momentum at the end of the dynamics, and
         'momentum_std' the standard deviation of the Gaussian distribution the
         momentum is drawn from
        :param y_2: value of the target function at the end of the dynamics
        :return: tuple with the accepted observation and target function
         evaluation
        """

        kin_energy_1 = self.kinetic_energy(
            dynamics_output['momentum_1'], dynamics_output['momentum_std']
        )
        kin_energy_2 = self.kinetic_energy(
            dynamics_output['momentum_2'], dynamics_output['momentum_std']
        )
        metropolis_hastings_ratio = (
            np.exp(self.init_y - y_2 + kin_energy_1 - kin_energy_2)
        )
        random_nbr = np.random.rand(1)[0]
        if random_nbr < metropolis_hastings_ratio:
            return dynamics_output['dynamics_end_obs'], y_2
        else:
            return self.init_obs, self.init_y

    def add_observation(self, obs):
        """Add observation & target function evaluation

        :param obs: tupple with size dimension
        :return: evaluation of the target function at obs
        """

        assert len(obs) == self.dimension

        self._list_obs.append(tuple(obs))
        y_end = self.target_function(obs)
        self._list_y.append(y_end)

        return y_end

    def define_gp(self, maxiter=1000, disp=True):
        """Instantiate the Gaussian process with the best covariance param"""

        self._fit_covariance_parameters(maxiter=maxiter, disp=disp)
        self.gp = self.likelihood_optimizer.instanciate_gp(
            self.current_cov_param
        )
        self.gp.covariance_matrix()

    def exploration(self, epsilon=0.1, length=100, gp_update_rate=4,
                    momentum_std=1.0):
        """Exploration phase

        The new observations and target function evaluations are added to
        _list_obs and _list_y

        :param epsilon: time step of the Hamiltonian dynamics
        :param length: length of the Hamiltonian dynamics
        :param gp_update_rate: int, rate at wich the Gaussian process
         covariance parameters are updated
        :param momentum_std: standard deviation of the Gaussian distribution
         the momentum is drawn from
        """

        self.define_gp(disp=False)
        self.init_y = np.min(self._list_y)
        self.init_obs = self._list_obs[np.argmin(self._list_y)]
        for idx in tqdm(range(self.n_explo)):
            self._update_current_sample(
                'exploration', epsilon=epsilon, length=length,
                momentum_std=momentum_std
            )
            if idx % gp_update_rate == 0:
                self.define_gp(disp=False)

    @staticmethod
    def kinetic_energy(momentum, std):
        """Kinetic energy

        :param momentum: fictitious momentum
        :param std: standard deviation of the Gaussian distribution the
         momentum is drawn from
        :return: kinetic energy corresponding to the momentum
        """

        energy = np.sum(np.power(momentum/std, 2))/2

        return energy

    @property
    def list_y(self):
        """List of target function evaluation"""

        return deepcopy(self._list_y)

    @property
    def list_obs(self):
        """List of observation where the target function has been evaluated"""

        return deepcopy(self._list_obs)

    def sample(self, epsilon=0.1, length=100, momentum_std=1.0):
        """Sampling phase

        This is a generator which will generate new samples according to the
        state of the sampler and the input parameters. The new observations and
        target function evaluations will be added to _list_obs and _list_y

        :param epsilon: time step of the Hamiltonian dynamics
        :param length: length of the Hamiltonian dynamics
        :param momentum_std: standard deviation of the Gaussian distribution
         the momentum is drawn from
        """

        if not (self.init_obs and self.init_y and self.gp):
            raise NoExplorationError

        self.init_y = np.min(self._list_y)
        self.init_obs = self._list_obs[np.argmin(self._list_y)]

        while True:
            self._update_current_sample(
                'sampling', epsilon=epsilon, length=length,
                momentum_std=momentum_std
            )
            yield self.init_obs
