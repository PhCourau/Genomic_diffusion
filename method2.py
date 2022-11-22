import numpy as np
import matplotlib.pyplot as plt

PRECISION = 1001 # Precision of the Fourier transform
                 #(should be odd to avoid the Nyquist frequency)
if PRECISION%2==0:
    Warning("PRECISION should be odd")
                 
def switch_around(M):
    """Switches terms of an fft-output to get a hermitian matrix
    Note that this operator is its own inverse"""
    n = M.shape[0]
    for k in range((n+1)//2):
        for i in range(n):
            M[k,i],M[-k,i] = M[-k,i],M[k,i]
    return M

def generate_mu():
    """An example of mean measure"""
    return [0]*PRECISION

def generate_covf():
    """An example of covariance measure"""
    covf = np.zeros([PRECISION,PRECISION])
    for x in range(PRECISION//2):
        for y in range(PRECISION//2):
            covf[x,y]=-0.01
            covf[x+(PRECISION+1)//2,y+(PRECISION+1)//2]=-1
    return covf

def make_diffusion_from_matrix(covf,mu=generate_mu(),sigma=1):
    """Makes a random diffusion from a function matrix and a mean matrix

    Parameters
    ----------
    covf : array_like
        Represents a symmetric bimeasure on [0,1]. Gives the covariances
        between small regions. Should have shape (PRECISION,PRECISION)
    mu : 1-d array, optional
        Represents a measure on [0,1]. Gives the mean increase of the process.
    sigma: float
        A multiplicative constant
    Returns
    -------
    X : nd_array
        A random diffusion on [0,1]. It verifies:
    E[X(t)X(s)] = min(t,s) + integral(f(t',s'),t' in [0,t],s' in [0,s])
    """
    # 1) Get the matrix M corresponding to the Fourier transform of covf
    M = np.fft.fft2(covf,norm="ortho")
    #The fft2 return is not hermitian, so we have to switch the terms 
    M = switch_around(M)
    M = M+np.identity(PRECISION)

    # 2) Diagonalise the matrix M and generate random variables
    eigval,eigvec = np.linalg.eig(M)
    random_variables = np.random.randn(PRECISION)*np.sqrt(eigval+0j)
    frequencies = eigvec @ random_variables

    #3) Invert Fourier transform
    pointwise_diffusion = np.fft.ifft(frequencies,norm="ortho")
    if (np.sum(np.abs(np.imag(pointwise_diffusion)))
        > 0.1*np.sum(np.abs(np.real(pointwise_diffusion)))):
        Warning("Losing a non-negligible imaginary part")
    pointwise_diffusion = np.real(pointwise_diffusion)
    return sigma*np.cumsum(pointwise_diffusion + mu)/np.sqrt(PRECISION)

def fragment(partition,r):
    """Fragments every block of a partition according to a Poisson Point
    process with intensity r

    Parameters
    ----------
    partition : list of lists of sorted integers
    r : float
        The probability of recombination
    """
    new_blocks = []
    indexes_to_remove = []
    for (block_index,block) in enumerate(partition):
        if np.random.random()<r*(block[-1]-block[0])/PRECISION:
            breakpoint = np.random.randint(block[0],block[-1]+1)
            
            newblock_even,newblock_odd = (
                [k for k in range(block[0],breakpoint+1)],
                [k for k in range(breakpoint+1,block[-1]+1)]
            )
            indexes_to_remove = [block_index] + indexes_to_remove
            new_blocks += [newblock_even,newblock_odd]
    for i in indexes_to_remove:
        partition.pop(i)
    return partition + new_blocks


def generate_random_cov(r=1,xi=0.01,theta=0.1,eps=1e-2):
    """A random covariance matrix generated according to the co-evolution
    process, with precision eps.

    Parameters
    ----------
    xi : The strength of selection
    theta : The intensity of mutation. Should be larger than xi
    r : The recombination rate. Should be larger than 1/PRECISION
    eps : The precision. Should be small relative to 1-theta.
    """
    covf = np.zeros([PRECISION,PRECISION])
    Tmax = int(np.log(eps*theta)/(np.log(1-theta)))+1#number of iterations
    partition = [[i for i in range(PRECISION)]]

    for t in range(Tmax):
        partition = fragment(partition,r)
        for block in partition:
            for i in range(len(block)):
                for j in range(i):
                    covf[block[i],block[j]] += -xi*(1-theta)**(2*(t+1))
                    covf[block[j],block[i]] += -xi*(1-theta)**(2*(t+1))
                covf[block[i],block[i]] += -xi*(1-theta)**(2*(t+1))
    return covf

        
        
