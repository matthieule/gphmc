## Gaussian Process Hamiltonian Monte Carlo

Implementation of Gaussian Process Hamiltonian Monte Carlo described in [2].
Some notable differences with the method described in the paper are:
- we do not use the gradient information at the evaluation point. As such, 
we do not need to know how to compute the derivative of the target density
- we use a black box optimization algorithm for the likelihood optimization
of the covariance parameters. As such, we do not need to define the gradient
of the likelihood with respect to the covariance parameters.

The code is mainly based on:
- `GPHMCSampler`: the class used to sample from a probability distribution defined up to a factor
- `gaussian_process_regression`: a submodule for Gaussian process regression (or interpolation)

A script example scripts is in the `script` folder

## Using the Code

Example usage can be found in `script/example_2d.py`. One can define a new `potential_energy` to try it for their own function.
As noted in the original paper [2], the parameter `epsilon` for the exploration and sampling phase might need to be tuned.

One can run the example with:

```python
python script/example_2d.py 
```

## 2D Example

We replicate the 2D example from [2]. The goal is to sample 2D points from 
the following density (defined up to a normalization factor):

<img src="https://github.com/matthieule/gphmc/blob/master/figures/density.png" alt="alt text" width=500px>

Below is the result of the density interpolation after the exploration of the Gaussian
Process Hamiltonian Monte Carlo using 50 initialization points, and 50 explorations points:

<img src="https://github.com/matthieule/gphmc/blob/master/figures/estimation.png" alt="alt text" width=500px>

The scatter points are the data points used to interpolate the target density on the 2D space, overlaid on the said interpolation.

Below is the results from the sampling stage of the Gaussian Process Hamiltonian Monte Carlo:

<img src="https://github.com/matthieule/gphmc/blob/master/figures/sample.png" alt="alt text" width=500px>

The pictures can be re-generated using:

```python
python script/example_2d.py 
```

## References:
- [1] [Rasmussen, C.E., 2006. Gaussian processes for machine learning.](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
- [2] [Rasmussen, C.E., Bernardo, J.M., Bayarri, M.J., Berger, J.O., Dawid, A.P., Heckerman, D., Smith, A.F.M. and West, M., 2003. Gaussian processes to speed up hybrid Monte Carlo for expensive Bayesian integrals. In Bayesian Statistics 7 (pp. 651-659).](http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/pdfs/pdf2080.pdf)
- [3] https://arogozhnikov.github.io/2016/12/19/markov_chain_monte_carlo.html
