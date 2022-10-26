import numpy as np
import matplotlib.pyplot as plt

PRECISION = 10000

def generate_brownian(t=1,size=PRECISION):
    """Generates PRECISION steps of a Brownian motion over time t"""
    steps = np.random.binomial(n=1,
                               size=size,
                               p=0.5)
    return np.cumsum(2*steps-1)*np.sqrt(t/PRECISION)

def generate_bridge(t=1,alpha=.1,gamma=1,size=PRECISION):
    a= generate_brownian(t,size)
    return gamma*a - alpha*np.linspace(0,t,size)*a[-1]

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
        nb_breakpoints = np.random.poisson(r*(block[-1]-block[0])/PRECISION)
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
        y1 = generate_bridge(t=block[-1]/PRECISION,
                             alpha=alpha,
                             gamma=gamma,
                             size=block[-1]+1)
        y1 -= (alpha*np.linspace(0,
                                 (block[-1]+1)/PRECISION,
                                 block[-1]+1)
                 *np.random.normal(loc=0,
                                   scale=np.sqrt(1-(block[-1]+1)
                                                 /PRECISION)))
        y2 = np.zeros(PRECISION)
        blockindex = 0
        nextindex = block[blockindex]
        if nextindex==0:
            y2[0] = y1[0]
            blockindex +=1
            nextindex = block[blockindex]
        for k in range(1,PRECISION):
            if k==nextindex:
                y2[k] = y2[k-1]+y1[nextindex]-y1[nextindex-1]
                blockindex +=1
                if blockindex<len(block):
                    nextindex = block[blockindex]
            else:
                y2[k]=y2[k-1]
        y0 += y2
    return y0

def generate_genomic_diffusion(r,xi,theta,eps=.01,sigma=1,checkmu=0):
    """Generates a genomic diffusion.
    Parameters
    ----------
    r: float
        rate of recombination
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

def plot_bridges(t=1,alpha=.4,intervals=None):
    """Will cut [0,1] into intervals and plot Brownian bridges in evory one.
    Just an illustration"""
    if intervals is None:
        intervals = [[k for k in range(PRECISION//4)]
                     +[k for k in range(PRECISION//2,(3*PRECISION)//4)],
                     [k for k in range(PRECISION//4,PRECISION//2)],
                     [k for k in range((3*PRECISION)//4,PRECISION)]]

    x=np.linspace(0,1,PRECISION)
    bridges=[]
    for k in range(len(intervals)):
        bridges.append(generate_bridge(t=len(intervals[k])/PRECISION,
                                       alpha=alpha,
                                       size=len(intervals[k])))
    z = list(bridges[0][:PRECISION//4])
    z += list(bridges[1]+z[PRECISION//4-1])
    z += list(bridges[0][PRECISION//4:]
              +z[PRECISION//2-1]
              -bridges[0][PRECISION//4-1])
    z += list(bridges[2]+z[(3*PRECISION)//4-1])
    plt.figure(figsize=[100,100])
    plt.plot(x[:PRECISION//4],
             z[:PRECISION//4],color="blue")
    plt.plot(x[PRECISION//2: (3*PRECISION)//4],
             z[PRECISION//2:(3*PRECISION)//4],color="blue")
    plt.plot([x[k] for k in intervals[1]],
             [z[k] for k in intervals[1]],color="red")
    plt.plot([x[k] for k in intervals[2]],
             [z[k] for k in intervals[2]],color="green")
    for breakpoint in [PRECISION//4,PRECISION//2,(3*PRECISION)//4]:
        plt.plot([x[breakpoint]]*2,
                 [z[breakpoint]-.5,
                  z[breakpoint]+.5],
                 color="black")
    plt.show()
