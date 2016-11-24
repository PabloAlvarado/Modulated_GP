import numpy as np
from matplotlib import pyplot as plt
import modgp
import GPflow

#%%
def logistic(x):
    return 1./(1+ np.exp(-x))

#%%    
X = np.linspace(0, 1, 2000).reshape(-1, 1)
k1 = GPflow.kernels.Matern12(input_dim=1, lengthscales=0.01)
k2 = GPflow.kernels.Matern52(input_dim=1, lengthscales=0.1, variance=10)

#%%
K1 = k1.compute_K_symm(X)
K2 = k2.compute_K_symm(X)

#%%
noise_var = 0.001
np.random.seed(1)
f = np.random.multivariate_normal(np.zeros(X.shape[0]), K1).reshape(-1, 1)
g = np.random.multivariate_normal(np.zeros(X.shape[0]), K2).reshape(-1, 1)
mean = f * logistic(g)
y = mean + np.random.randn(*mean.shape) * np.sqrt(noise_var)

#%%
plt.plot(X, f, 'b')
plt.plot(X, logistic(g), 'g', lw=2)
plt.figure()
plt.plot(X, y, 'b')

#%%
m = modgp.ModGP(X, y, k1, k2, X[::8].copy())

#%%
m.kern1.fixed = True
m.kern2.fixed = True
m.likelihood.noise_var = noise_var
m.likelihood.noise_var.fixed = True

#%%
m.optimize(disp=1,maxiter=10)

#%%
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

