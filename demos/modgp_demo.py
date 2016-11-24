import numpy as np
from matplotlib import pyplot as plt
import modgp
import GPflow

#%% 
def logistic(x):
    return 1./(1+ np.exp(-x))

#%%   
X = np.linspace(0, 1, 2000).reshape(-1, 1) # (-1, 1) means any number of necessay rows and  1 column
k1 = GPflow.kernels.Matern12(input_dim=1, lengthscales=0.01)
k2 = GPflow.kernels.Matern52(input_dim=1, lengthscales=0.1, variance=10.)

#%% calculate cov matrices
K1 = k1.compute_K_symm(X)
K2 = k2.compute_K_symm(X)

#%% sample f and g, transform g and generate the "observed" data y.
noise_var = 0.001
np.random.seed(1)
f = np.random.multivariate_normal(np.zeros(X.shape[0]), K1).reshape(-1, 1)
g = np.random.multivariate_normal(np.zeros(X.shape[0]), K2).reshape(-1, 1)
mean = f * logistic(g)
y = mean + np.random.randn(*mean.shape) * np.sqrt(noise_var)

#%% plot the latent functions f(t) and \sigma(g(t)), plot the observed variable y(t)
plt.figure()
plt.plot(X, f, 'b')
plt.plot(X, logistic(g), 'g', lw=2)
plt.figure()
plt.plot(X, y, 'b')

#%% generate model object
Z = X[::8].copy() # copy inducting points from input vector X
m = modgp.ModGP(X, y, k1, k2, Z) # X -> input variable, y -> observed data, k1,k2 -> kernels, Z -> inducting points

#%% keep model parameters fixed.
m.kern1.fixed = True
m.kern2.fixed = True
m.likelihood.noise_var = noise_var
m.likelihood.noise_var.fixed = True

#%% optimize the approximated distributions over f(t) and g(t).
m.optimize(disp=1,maxiter=10)

#%% evaluate predictions
mu, var = m.predict_f(X)
plt.plot(X, mu, 'b')
plt.plot(X, mu + 2*np.sqrt(var), 'b--')
plt.plot(X, mu - 2*np.sqrt(var), 'b--')

plt.twinx()
mu, var = m.predict_g(X)
plt.plot(X, logistic(mu), 'g')
plt.plot(X, logistic(mu + 2*np.sqrt(var)), 'g--')
plt.plot(X, logistic(mu - 2*np.sqrt(var)), 'g--')

#%%

