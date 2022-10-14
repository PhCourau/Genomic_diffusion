import numpy as np
import matplotlib.pyplot as plt

NB_POINTS = 1001 # Precision of the Fourier transform
                 #(should be odd to avoid the Nyquist frequency)
if NB_POINTS%2==0:
    Warning("NB_POINTS should be odd")
                 
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
    return [0]*NB_POINTS

def generate_covf():
    """An example of covariance measure"""
    covf = np.zeros([NB_POINTS,NB_POINTS])
    for x in range(NB_POINTS//2):
        for y in range(NB_POINTS//2):
            covf[x,y]=-0.01
            covf[x+(NB_POINTS+1)//2,y+(NB_POINTS+1)//2]=-1
    return covf

def make_diffusion_from_matrix(covf,mu=generate_mu()):
    """Makes a random diffusion from a function matrix and a mean matrix

    Parameters
    ----------
    covf : array_like
        Represents a symmetric bimeasure on [0,1]. Gives the covariances
        between small regions. Should have shape (NB_POINTS,NB_POINTS)
    mu : 1-d array, optional
        Represents a measure on [0,1]. Gives the mean increase of the process.

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
    M = M+np.identity(NB_POINTS)

    # 2) Diagonalise the matrix M and generate random variables
    eigval,eigvec = np.linalg.eig(M)
    random_variables = np.random.randn(NB_POINTS)*np.sqrt(eigval+0j)
    frequencies = eigvec @ random_variables

    #3) Invert Fourier transform
    pointwise_diffusion = np.fft.ifft(frequencies,norm="ortho")
    if (np.sum(np.abs(np.imag(pointwise_diffusion)))
        > 0.1*np.sum(np.abs(np.real(pointwise_diffusion)))):
        Warning("Losing a non-negligible imaginary part")
    pointwise_diffusion = np.real(pointwise_diffusion)
    return np.cumsum(pointwise_diffusion + mu)/NB_POINTS

def fragment(partition,r):
    """Fragments every block of a partition according to a Poisson Point
    process with intensity r

    Parameters
    ----------
    partition : list of lists of sorted integers
    r : float
        The intensity of recombination
    """
    new_blocks = []
    indexes_to_remove = []
    for (block_index,block) in enumerate(partition):
        nb_breakpoints = np.random.poisson(r*(block[-1]-block[0])/NB_POINTS)
        if nb_breakpoints==0:
            return partition

        breakpoints = np.random.randint(block[0],block[-1]+1,nb_breakpoints)
        breakpoints.sort()
        breakpoints = np.append(breakpoints,block[-1]+1)

        newblock_even,newblock_odd = [block[0]],[]
        are_we_odd = False
        last_element = block[0]
        breakindex = 0
        for element in block[1:]:
            while last_element < breakpoints[breakindex] <= element:
                are_we_odd = not are_we_odd
                breakindex += 1
            if are_we_odd:
                newblock_odd += [element]
            else:
                newblock_even += [element]
            last_element = element
        if min(len(newblock_odd),len(newblock_even)) > 0:
            indexes_to_remove = [block_index] + indexes_to_remove
            new_blocks += [newblock_even,newblock_odd]
    for i in indexes_to_remove:
        partition.pop(i)
    return partition + new_blocks

def generate_random_cov(xi=0.01,theta=0.1,r=1,eps=0.1):
    """A random covariance matrix generated according to the co-evolution
    process, with precision eps.

    Parameters
    ----------
    xi : The strength of selection
    theta : The intensity of mutation. Should be larger than xi
    r : The recombination rate. Should be larger than 1/NB_POINTS
    eps : The precision. Should be small relative to 1-theta.
    """
    covf = np.zeros([NB_POINTS,NB_POINTS])
    keps = int(np.log(eps*theta)/(2*np.log(1-theta)))+1#number of iterations
    partition = [[i for i in range(NB_POINTS)]]

    for k in range(keps):
        partition = fragment(partition,r)
        for block in partition:
            for i in range(len(block)):
                for j in range(i):
                    covf[block[i],block[j]] += -xi*(1-theta)**(2*k)
                    covf[block[j],block[i]] += -xi*(1-theta)**(2*k)
                covf[block[i],block[i]] += -xi*(1-theta)**(2*k)
    return covf

        
        
