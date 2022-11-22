import numpy as np
import matplotlib.pyplot as plt

PRECISION = 1001

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

def generate_concatenated_bridges(partition,alpha,gamma):
    """Generates a concatenation of bridges according to a partition
    Parameters
    ----------
    partition: list of lists of ints
        The blocks correspond to the positions of the bridges.
        Should be a partition of range(PRECISION)
    alpha: float
        parameter (see paper)
    gamma: float
        parameter (see paper)
         """
    y0 = np.zeros(PRECISION)
    for block in partition:
        #generate y_x^{k,t}
        y1 = np.random.binomial(p=1/2,
                                n=1,
                                size=block[-1]-block[0])
        y1 = (2*y1-1)/np.sqrt(PRECISION)
        y1 = gamma*y1 - alpha*(np.sum(y1)
                    +np.random.normal(0,1-(block[-1]-block[0])/PRECISION)
                               )
        y0[block[0]:block[-1]+1] += y1
    return np.cumsum(y0)

def generate_genomic_diffusion(r,xi,theta,eps=.01,sigma=1,checkmu=0):
    """Generates a genomic diffusion.
    Parameters
    ----------
    r: float
        rate of recombination (0<=r<=1)
    xi: float
        strength of selection
    theta: float
        rate of mutation (should be greater than xi)
    eps: float
        precision
    sigma: float
        mutational variance
    checkmu: float
        mutational mean
    """
    Tmax = int(np.log(eps*theta)/np.log(1-theta))
    gamma = np.sqrt(theta*(2-theta))/(1-theta)
    alpha = gamma - np.sqrt(gamma**2-xi)
    frag =[[k for k in range(PRECISION)]]#Will be sequentially fragmented
    z = np.zeros(PRECISION)
    for t in range(1,Tmax):
        frag = fragment(frag,r)
        z+=(1-theta)**t * generate_concatenated_bridges(frag,alpha,gamma)
    return sigma*z + np.linspace(0,1,PRECISION)*(
        1-(1-theta)/theta*xi)*checkmu


