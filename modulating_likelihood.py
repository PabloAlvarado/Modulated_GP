import GPflow 
import tensorflow as tf
import numpy as np
import itertools #The module standardizes a core set of fast, memory efficient tools.


def mvhermgauss(means, covs, H, D):
        """
        Return the evaluation locations, and weights for several multivariate Hermite-Gauss quadrature runs.
        :param means: NxD
        :param covs: NxDxD
        :param H: Number of Gauss-Hermite evaluation points.
        :param D: Number of input dimensions. Needs to be known at call-time.
        :return: eval_locations (H**DxNxD), weights (H**D)
        """
        N = tf.shape(means)[0] 
        gh_x, gh_w = GPflow.likelihoods.hermgauss(H) # calculate H HerGaus-cuadrature evaluation points and its corresponding weights
        xn = np.array(list(itertools.product(*(gh_x,) * D)))  # H**DxD #expand 1 dimensional grid (evaluation points) to D dimensions. 
        wn = np.prod(np.array(list(itertools.product(*(gh_w,) * D))), 1)  # H**D #expand weights to the new grid
        cholXcov = tf.cholesky(covs)  # NxDxD #get cholesky decomposition of the cov matrix
        X = 2.0 ** 0.5 * tf.batch_matmul(cholXcov, tf.tile(xn[None, :, :], (N, 1, 1)), adj_y=True) + tf.expand_dims(means, 2)#NxDxH**D
        Xr = tf.reshape(tf.transpose(X, [2, 0, 1]), (-1, D))  # H**DxNxD
        # the 2 lines above rescale the grid xn taking into acount the mean vector and cov matrix of the Gaussian distribution
        return Xr, wn * np.pi ** (-D * 0.5) # return the rescaled weights and evaluation locations.


class ModLik(GPflow.likelihoods.Likelihood):
    def __init__(self):
        GPflow.likelihoods.Likelihood.__init__(self)
        self.noise_var = GPflow.param.Param(1.0)

    def logp(self, F, Y):
        f, g, h = F[:, 0], F[:, 1], F[:,2]
        y = Y[:, 0]
        sigma_g = 1./(1 + tf.exp(-g))  # squash g to be positive
        mean = f * sigma_g + h
        return GPflow.densities.gaussian(y, mean, self.noise_var).reshape(-1, 1)

    def variational_expectations(self, Fmu, Fvar, Y):
        D = 3  # Number of input dimensions
        H = 10 # number of Gauss-Hermite evaluation points.
        Xr, w = mvhermgauss(Fmu, tf.matrix_diag(Fvar), H, D)
        w = tf.reshape(w, [-1, 1])
        f, g, h = Xr[:, 0], Xr[:, 1], Xr[:, 2]
        y = tf.tile(Y, [H**D, 1])[:, 0]
        sigma_g = 1./(1 + tf.exp(-g))  # squash g to be positive
        mean = f * sigma_g + h
        evaluations = GPflow.densities.gaussian(y, mean, self.noise_var)
        evaluations = tf.transpose(tf.reshape(evaluations, tf.pack([tf.size(w), tf.shape(Fmu)[0]])))
        return tf.matmul(evaluations, w)
