import numpy as np
import numpy.matlib
from typing import List, Tuple, Dict, Optional
from scipy.linalg import null_space
from scipy.stats import norm, uniform, beta

XYZ = ['x', 'y', 'z']


def vec2rotmag(vect: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """
    Takes a vector and breaks it down into constituent
    Euler rotations and magnitude
    """
    assert all([k in vect.keys() for k in XYZ]),\
        "Ill formed vector provided: " + str(vect)
    x = vect['x']
    y = vect['y']
    z = vect['z']
    mag = np.sqrt(x*x + y*y + z*z)
    rot = dict([[k, vect[k]/mag] for k in XYZ])
    return (rot, mag)


def rotmag2vec(rotation: Dict[str, float],
               magnitude: float) -> Dict[str, float]:
    """
    Takes a rotation [x,y,z] angle dict and magnitude,
    and returns the equivalent [x,y,z] vector
    """
    assert all([k in rotation.keys() for k in XYZ]),\
        "Ill formed vector provided: " + str(rotation)
    return dict([[k, rotation[k]*magnitude] for k in XYZ])


def euler2vec(euler: Dict[str, float]) -> Dict[str, float]:
    """
    Takes a euler rotation [x,y,z] angle dict,
    and returns the equivalent angles to the [x,y,z] axes
    """
    assert all([k in euler.keys() for k in XYZ]),\
        "Ill formed vector provided: " + str(euler)
    return dict([['x', np.cos(euler['x']*np.cos(euler['y']))], 
                 ['y', np.sin(euler['x']*np.cos(euler['y']))], 
                 ['z', np.sin(euler['y'])]])


def rad2deg(rad: float) -> float:
    return rad * 180 / np.pi


def deg2rad(deg: float) -> float:
    return deg / 180 * np.pi

def rand_uniform_hypersphere(seed, N,p):

    """
        rand_uniform_hypersphere(N,p)
        =============================
        Generate random samples from the uniform distribution on the (p-1)-dimensional
        hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$. We use the method by
        Muller [1], see also Ref. [2] for other methods.
        INPUT:  
            * N (int) - Number of samples
            * p (int) - The dimension of the generated samples on the (p-1)-dimensional hypersphere.
                - p = 2 for the unit circle $\mathbb{S}^{1}$
                - p = 3 for the unit sphere $\mathbb{S}^{2}$
            Note that the (p-1)-dimensional hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$ and the
            samples are unit vectors in $\mathbb{R}^{p}$ that lie on the sphere $\mathbb{S}^{p-1}$.
    References:
    [1] Muller, M. E. "A Note on a Method for Generating Points Uniformly on N-Dimensional Spheres."
    Comm. Assoc. Comput. Mach. 2, 19-20, Apr. 1959.
    [2] https://mathworld.wolfram.com/SpherePointPicking.html
    """

    if (p<=0) or (type(p) is not int):
        raise Exception("p must be a positive integer.")

    # Check N>0 and is an int
    if (N<=0) or (type(N) is not int):
        raise Exception("N must be a non-zero positive integer.")

    # v = np.random.normal(0,1,(N,p))
    v = norm.rvs(0, 1, (N,p), random_state=seed)
    v = np.divide(v,np.linalg.norm(v,axis=1,keepdims=True))

    return v


def rand_t_marginal(seed, kappa,p,N=1):
    """
        rand_t_marginal(kappa,p,N=1)
        ============================
        Samples the marginal distribution of t using rejection sampling of Wood [3].
        INPUT:
            * kappa (float) - concentration        
            * p (int) - The dimension of the generated samples on the (p-1)-dimensional hypersphere.
                - p = 2 for the unit circle $\mathbb{S}^{1}$
                - p = 3 for the unit sphere $\mathbb{S}^{2}$
            Note that the (p-1)-dimensional hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$ and the
            samples are unit vectors in $\mathbb{R}^{p}$ that lie on the sphere $\mathbb{S}^{p-1}$.
            * N (int) - number of samples
        OUTPUT:
            * samples (array of floats of shape (N,1)) - samples of the marginal distribution of t
    """

    # Check kappa >= 0 is numeric
    if (kappa < 0) or ((type(kappa) is not float) and (type(kappa) is not int)):
        raise Exception("kappa must be a non-negative number.")

    if (p<=0) or (type(p) is not int):
        raise Exception("p must be a positive integer.")

    # Check N>0 and is an int
    if (N<=0) or (type(N) is not int):
        raise Exception("N must be a non-zero positive integer.")


    # Start of algorithm
    b = (p - 1.0) / (2.0 * kappa + np.sqrt(4.0 * kappa**2 + (p - 1.0)**2 ))    
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (p - 1.0) * np.log(1.0 - x0**2)

    samples = np.zeros((N,1))

    # Loop over number of samples
    for i in range(N):

        # Continue unil you have an acceptable sample
        while True:

            # Sample Beta distribution
            # Z = np.random.beta( (p - 1.0)/2.0, (p - 1.0)/2.0 )
            Z = beta.rvs((p - 1.0)/2.0, (p - 1.0)/2.0, random_state=seed)

            # Sample Uniform distribution
            # U = np.random.uniform(low=0.0,high=1.0)
            U = uniform.rvs(0.0, 1.0, random_state=seed)

            # W is essentially t
            W = (1.0 - (1.0 + b) * Z) / (1.0 - (1.0 - b) * Z)

            # Check whether to accept or reject
            if kappa * W + (p - 1.0)*np.log(1.0 - x0*W) - c >= np.log(U):

                # Accept sample
                samples[i] = W
                break
            seed += 1

    return samples


def rand_von_mises_fisher(seed, mu,kappa,N=1):
    """
        rand_von_mises_fisher(mu,kappa,N=1)
        ===================================
        Samples the von Mises-Fisher distribution with mean direction mu and concentration kappa.
        INPUT:
            * mu (array of floats of shape (p,1)) - mean direction. This should be a unit vector.
            * kappa (float) - concentration.
            * N (int) - Number of samples.
        OUTPUT:
            * samples (array of floats of shape (N,p)) - samples of the von Mises-Fisher distribution
            with mean direction mu and concentration kappa.
    """


    # Check that mu is a unit vector
    eps = 10**(-8) # Precision
    norm_mu = np.linalg.norm(mu)
    if abs(norm_mu - 1.0) > eps:
        raise Exception("mu must be a unit vector.")

    # Check kappa >= 0 is numeric
    if (kappa < 0) or ((type(kappa) is not float) and (type(kappa) is not int)):
        raise Exception("kappa must be a non-negative number.")

    # Check N>0 and is an int
    if (N<=0) or (type(N) is not int):
        raise Exception("N must be a non-zero positive integer.")

    # Dimension p
    p = len(mu)

    # Make sure that mu has a shape of px1
    mu = np.reshape(mu,(p,1))

    # Array to store samples
    samples = np.zeros((N,p))

    #  Component in the direction of mu (Nx1)
    t = rand_t_marginal(seed, kappa,p,N)

    # Component orthogonal to mu (Nx(p-1))
    xi = rand_uniform_hypersphere(seed, N,p-1)

    # von-Mises-Fisher samples Nxp

    # Component in the direction of mu (Nx1).
    # Note that here we are choosing an
    # intermediate mu = [1, 0, 0, 0, ..., 0] later
    # we rotate to the desired mu below
    samples[:,[0]] = t

    # Component orthogonal to mu (Nx(p-1))
    samples[:,1:] = np.matlib.repmat(np.sqrt(1 - t**2), 1, p-1) * xi

    # Rotation of samples to desired mu
    O = null_space(mu.T)
    R = np.concatenate((mu,O),axis=1)
    samples = np.dot(R,samples.T).T

    return samples