import numpy as np
# for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# Feature Selection algorithm that is being used
from sklearn.feature_selection import mutual_info_classif
# May be helpful for the NP test 
from scipy.stats import binom
from scipy.stats import chi2
from scipy.stats import norm
from math import comb

def my_NP_test(NP_in, alpha=0.01):
    """This function needs to be correctly implemented by you.
    It performs the Neyman-Pearson test on the data from the Feature Selection algorithm, as described in the paper
    "A Bootstrap Based Neyman-Pearson Test for Identifying Variable Importance," 
    by G. Ditzler, R. Polikar, and G. Rosen, IEEE Transactions on Neural Networks and Learning Systems, April 2015.
    
    The function accepts 
        NP_in   -   is a 0-1 matrix with values 1 for the indeces of features selected by the Feature Selection algorithm in the n_boots tests
                    it is a numpy array of size n_features x n_boots
                    (it is the matrix X in the paper)
        alpha   -   is the false alarm probability of the NP test          
                    if not provided as an input, its default value is 0.01
    The function returns
        NP_out  -   is a 0-1 vector with values 1 for the indeces of features selected by the NP tests as being significant
                    it is a numpy array of size n_features."""
    
    # Below is where you need to fill in your code.
    # Replace the two lines below, which are only there to make the current program execute.
    n_features, n_boots = NP_in.shape
    NP_out = np.zeros(n_features)
    sumup = 0
    crit = 0
    z = np.arange(0, n_boots, 1)
    selected_features = 0
    for i in range(n_features):
        if(NP_in[i][0]) == 1:
            selected_features += 1

    p0 = selected_features / n_features
    for i, zeta in enumerate(z):
        sumup += comb(n_boots, zeta) * (p0**zeta) * ((1 - p0)**(n_boots - zeta))
        if (1 - sumup) < alpha:
            crit = zeta
    for i in range(n_features):
        V = NP_in[i, :]
        if sum(V) > crit:
            NP_out[i] = 1
    return NP_out

def test1():
    """This function performs the example using synthetic data leading to results shown in Figure 1 of the paper
    "A Bootstrap Based Neyman-Pearson Test for Identifying Variable Importance," 
    by G. Ditzler, R. Polikar, and G. Rosen, IEEE Transactions on Neural Networks and Learning Systems, April 2015.
    """
    
    # set the parameters for the experiment 
    n_features = 25         # parameter K in the paper
    n_observations = 1000   # parameter M in the paper
    n_relevant = 5          # parameter k* in the paper

    # we conduct the first synthetic data test from the paper
    
    # generate synthetic data     
    data = np.around(10*np.random.rand(n_observations, n_features));        # variable x in the paper
    label_sum = np.sum(data[:,:n_relevant],axis=1,dtype=np.int8)
    delta = 5*n_relevant;
    labels = np.zeros(n_observations,dtype=np.int8);
    labels[label_sum <= delta] = 1;                                         # variable y in the paper

    # parameters for NP test
    n_select = 10           # parameter k in the paper 
    n_boots = 100           # parameter n in the paper
    alpha = 0.01            # parameter alpha in the paper 
    
    # run test
    # V is the 0-1 n_features x n_boots matrix returned by the Feature Selection algorithm
    # idx is the vector with the indices of the features that have been determined significant by the NP test 
    idx, V = npfs(data, labels, n_select, n_boots, alpha)
    # plot the result, similar to Figure 1 in the paper
    colors = [(0,0,0),(1,1,1),(1,0.5,0)]
    cmap = LinearSegmentedColormap.from_list('aa', colors, N=3)
    V[idx,:] = 2
    plt.figure()
    plt.imshow(V,cmap=cmap,origin='lower')
    plt.xlabel('bootstrap')
    plt.ylabel('feature')
    plt.savefig('test1.png')
def test2():
    """This function performs the example using synthetic data leading to results shown in Figure 2 of the paper
    "A Bootstrap Based Neyman-Pearson Test for Identifying Variable Importance," 
    by G. Ditzler, R. Polikar, and G. Rosen, IEEE Transactions on Neural Networks and Learning Systems, April 2015.
    """
    
     # set the parameters for the experiment 
    n_features = 50         # parameter K in the paper
    n_observations = 1000   # parameter M in the paper
    n_relevant = 15        # parameter k* in the paper

    # we conduct the second synthetic data test from the paper
    
    # generate synthetic data     
    data = np.around(10*np.random.rand(n_observations, n_features));        # variable x in the paper
    label_sum = np.sum(data[:,:n_relevant],axis=1,dtype=np.int8)
    delta = 5*n_relevant;
    labels = np.zeros(n_observations,dtype=np.int8);
    labels[label_sum <= delta] = 1;                                         # variable y in the paper

    # parameters for NP test
    n_select = 10           # parameter k in the paper 
    alpha = 0.01            # parameter alpha in the paper 
    
    # run test
    n_relevant_NP = []
    for n_boots in range(1,100,20):  # parameter n in the paper 
        # V is the 0-1 n_features x n_boots matrix returned by the Feature Selection algorithm
        # idx is the vector with the indices of the features that have been determined significant by the NP test 
        idx, V = npfs(data, labels, n_select, n_boots, alpha)
        # store the number of the statistically relevant features
        n_relevant_NP.append(idx[0].shape[0])
   
    # plot the result, similar to Figure 2 in the paper
    plt.figure()
    plt.plot(range(1,101,20),n_relevant_NP,'o-')
    plt.xlabel('number of bootstraps')
    plt.ylabel('# of features')
    plt.savefig('test2.png')
def npfs(data, labels, n_select=15, n_boots=100, alpha=0.01):
    """This function performs the feature selection using a Feature Selection algorithm 
        followed by the Neyman-Pearson test for the significance of selected features.
        
    The function accepts 
       data     -   is the collection of feature vectors, numpy array of size n_observations x n_features 
       labels   -   is the collection of labels, numpy array of size  n_observations 
       n_select -   is the number of features seleted by the Feature Selection algorithm
                    if not provided as an input, its default value is 15
       n_boots  -   is the number of bootstrap samples
                    if not provided as an input, its default value is 15
       alpha    -   is the false alarm probability of the NP test          
                    if not provided as an input, its default value is 0.01
    The function returns 
        idx     -   is the list of indeces of features that are significant according to the NP test, it is a tuple
        V       -   is a 0-1 matrix with values 1 for the indeces of features selected by the Feature Selection algorithm in the n_boots tests
                    it is a numpy array of size n_features x n_boots."""

    n_observations, n_features = data.shape
    V = np.zeros((n_features, n_boots),dtype=np.int8);
    
    # bootstrap sampling and Feature Selection algorithm
    bsns = (np.floor(0.75*n_observations)).astype('int32')
    for b in range(n_boots):
        ibs = np.random.choice(range(n_observations), bsns, replace=True, p=np.ones(n_observations)/n_observations)
        Xp = data[ibs]
        Yp = labels[ibs]
        f_score = mutual_info_classif(Xp, Yp, discrete_features=True);
        V[np.argsort(-f_score)[:n_select],b]=1
        
    # run NP test on output V of Feature Selection algorithm and with false alarm probability alpha
    NP_test = my_NP_test(V, alpha) 
    idx = np.nonzero(NP_test)
    
    return idx, V
    
