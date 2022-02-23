import types
import warnings
import numpy as np
from tqdm import tqdm
from sklearn.metrics import pairwise
from sklearn.cluster import AffinityPropagation


def pairwise_similarity(X: np.array, Y: np.array = None, affinity: str = 'cosine'):
    '''
    Compute affinity between samples in X and Y
    Args:
        X: Input data
        Y(object, optional, default=None): Input data. If None, the output will be the pairwise similarities between all samples in X
        affinity(object, optional, default='cosine'): the measure of similarity. Cosine Similarity is set as default, but you can pass your own function
    Returns:
        Returns similarity between samples in X and Y
    '''
    Y = X if Y is None else Y

    if affinity == 'cosine':
        return pairwise.cosine_similarity(X, Y)

    elif isinstance(affinity, types.FunctionType):
        # TODO: speed-up
        return np.array([[affinity(x, y) for y in Y] for x in tqdm(X, leave=True, position=0)])
    else:
        raise Exception('You must define a similarity measure')


class IncrementalAffinityPropagation:

    def __init__(self, damping:float=0.5, max_iter:int=200, convergence_iter:int=15,
                 preference:object=None, affinity:object='cosine', verbose:bool=False):
        '''Affinity Propagation Clustering of data.

        Args:
            preference(object, optional, default=None): array-like of shape (n_samples,) or float, default=None
                Preferences for each point - points with larger values of
                preferences are more likely to be chosen as exemplars. The number of
                exemplars, i.e. of clusters, is influenced by the input preferences
                value. If the preferences are not passed as arguments, they will be
                set to the median of the input similarities (resulting in a moderate
                number of clusters). For a smaller amount of clusters, this can be set
                to the minimum value of the similarities.
            damping(float, optional, default=0.5): damping factor
            affinity(object, optional, default=0.5): similarity measure for compute similarity. By default, cosine similarity. You can provide your own function.
            convergence_iter(int, optional, default=15): Number of iterations with no change in the number of estimated clusters that stops the convergence.
            max_iter(int, optional, default=200): Maximum number of iterations
            damping(float, optional, default=0.5): Damping factor between 0.5 and 1.
            verbose(bool, optional, default=False): The verbosity level.
        '''

        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.preference = preference
        self.verbose = verbose

        self.affinity = affinity

        # time step index of data arriving
        self.time_tag = 0

        # keep in memory data point
        self._X = None

        # keep in memory previous labels and clusters
        self.labels = dict()
        self.centroids = dict()

    def __step(self, ind:int, tmp) -> tuple:
        '''
        This is meant to return the new Availability (A) and Repsonsibility (R) matrices for all the data points

        Args:
            tmp(object): array-like of shape (n_samples, n_samples), Intermediate results
            ind(int): tmp index

        Returns:
            R(object): array-like of shape (n_samples, n_samples). Responsibility matrix
            A(object): array-like of shape (n_samples, n_samples). Availability matrix
        '''
        N, N = tmp.shape  # == self.R.shape == self.N.shape
        old_R, old_A = self.R, self.A

        # R UPDATE STEP - from sklearn
        np.add(old_A, self.S, tmp)
        I = np.argmax(tmp, axis=1)
        Y = tmp[ind, I]  # np.max(A + S, axis=1)
        tmp[ind, I] = -np.inf
        Y2 = np.max(tmp, axis=1)
        # tmp = Rnew
        np.subtract(self.S, Y[:, None], tmp)
        tmp[ind, I] = self.S[ind, I] - Y2

        # Damping
        tmp *= 1 - self.damping
        R = old_R * self.damping
        R += tmp

        # A UPDATE STEP - from sklearn
        # tmp = Rp; compute availabilities
        np.maximum(R, 0, tmp)
        tmp.flat[:: N + 1] = R.flat[:: N + 1]

        # tmp = -Anew
        tmp -= np.sum(tmp, axis=0)
        dA = np.diag(tmp).copy()
        tmp.clip(0, np.inf, tmp)
        tmp.flat[:: N + 1] = dA

        # Damping
        tmp *= 1 - self.damping
        A = old_A * self.damping
        A -= tmp

        return R, A

    def fit(self, X) -> tuple:
        '''
        This runs the Incremental Affinity Propagation Clustering Based on Nearest Neighbor Assignment until convergence

        Args:
            X(object): array-like of shape (n_samples, n_samples). Training instances to cluster

        Returns:
            exemplar : ndarray. Cluster exemplars
            labels : ndarray of shape (n_samples,). Cluster labels
        '''

        # first data
        if self.time_tag == 0:
            self._X = X

            # COMPUTE THE SIMILARITY MATRIX
            self.S = pairwise_similarity(X, affinity=self.affinity)

            if self.preference is None:
                self.preference = np.median(self.S)

            N, N = self.S.shape

            # Place preference on the diagonal of S
            self.preference = np.array(self.preference)
            self.S.flat[:: (N + 1)] = self.preference

            # Remove degeneracies
            self.S = self.S.reshape(N, N)

            #  INIITALISE THE RESPONSIBILITY AND THE AVAILABILITY MATRICES
            N, N = self.S.shape
            self.R, self.A = np.zeros((N, N)), np.zeros((N, N))

        # new data are available. Perform clustering from pre-computed matrices
        else:

            self._old_X = self._X

            # remove duplicates: data from timestep i that are already considered in timestep i-1
            if X.ndim == 1:
                new_X = np.setdiff1d(X, self._old_X)
            else:
                # TODO: remove duplicates
                new_X = X

            if len(new_X) == 0:
                warnings.warn("Incremental Affinity propagation needs new training instances.")
                return None, None

            self._X = np.concatenate([self._old_X, new_X])

            # Similarity matrix S update
            S_tmp1 = pairwise_similarity(new_X, self._old_X, affinity=self.affinity)
            S_tmp2 = pairwise_similarity(new_X, affinity=self.affinity)
            self.S = np.concatenate([np.concatenate([self.S, S_tmp1.T], axis=1),  # up & right
                                     np.concatenate([S_tmp1, S_tmp2], axis=1)])  # down
            if self.preference is None:
                self.preference = np.median(self.S)

            N, N = self.S.shape

            # Place preference on the diagonal of S
            #####self.preference = np.array(self.preference)
            #####self.S.flat[:: (N + 1)] = self.preference

            # Remove degeneracies
            #####self.S = self.S.reshape(N, N)

            # Responsibility matrix R update
            old_R, old_A, old_N = self.R, self.A, self.R.shape[1]

            R_tmp1 = np.array([[self.__matrix_update(self.S, old_R, i, j) for j in range(old_N)]
                               for i in range(old_N, N)]).T
            R_tmp2 = np.array([[self.__matrix_update(self.S, old_R, i, j) for i in range(old_N)]
                               for j in range(old_N, N)])
            R_tmp3 = np.zeros((new_X.shape[0], new_X.shape[0]))
            self.R = np.concatenate([np.concatenate([old_R, R_tmp1], axis=1), np.concatenate([R_tmp2, R_tmp3], axis=1)])

            # Availability matrix A update
            A_tmp1 = np.array([[self.__matrix_update(self.S, old_A, i, j) for j in range(old_N)]
                               for i in range(old_N, N)]).T
            A_tmp2 = np.array([[self.__matrix_update(self.S, old_A, i, j) for i in range(old_N)]
                               for j in range(old_N, N)])
            A_tmp3 = np.zeros((new_X.shape[0], new_X.shape[0]))
            self.A = np.concatenate([np.concatenate([old_A, A_tmp1], axis=1), np.concatenate([A_tmp2, A_tmp3], axis=1)])

        # -- #
        # Check for convergence (from scikit-learn)
        e = np.zeros((N, self.convergence_iter))
        # -- #

        ind = np.arange(N)
        tmp = np.zeros((N, N))

        for i in range(self.max_iter):
            if self.verbose:
                print("processing iteration %d" % (i,))
            self.R, self.A = self.__step(ind, tmp)

            # -- #
            # Check for convergence (from scikit-learn)
            E = (np.diag(self.A) + np.diag(self.R)) > 0
            e[:, i % self.convergence_iter] = E
            K = np.sum(E, axis=0)

            if i >= self.convergence_iter:
                se = np.sum(e, axis=1)
                unconverged = np.sum((se == self.convergence_iter) + (se == 0)) != N
                if (not unconverged and (K > 0)) or (i == self.max_iter):
                    never_converged = False
                    if self.verbose:
                        print("Converged after %d iterations." % i)
                    break
            # -- #

        else:
            never_converged = True
            if self.verbose:
                print("Did not converge")

        I = np.flatnonzero(E)
        K = I.size  # Identify exemplars

        if K > 0 and not never_converged:
            c = np.argmax(self.S[:, I], axis=1)
            c[I] = np.arange(K)  # Identify clusters
            # Refine the final set of exemplars and clusters and return results
            for k in range(K):
                ii = np.where(c == k)[0]
                j = np.argmax(np.sum(self.S[ii[:, np.newaxis], ii], axis=0))
                I[k] = ii[j]

            c = np.argmax(self.S[:, I], axis=1)
            c[I] = np.arange(K)
            labels = I[c]
            # Reduce labels to a sorted, gapless, list
            cluster_centers_indices = np.unique(labels)
            labels = np.searchsorted(cluster_centers_indices, labels)
        else:
            warnings.warn(
                "Affinity propagation did not converge, this model "
                "will not have any cluster centers.")
            labels = np.array([-1] * N)
            cluster_centers_indices = []

        self.labels[self.time_tag] = labels
        self.centroids[self.time_tag] = cluster_centers_indices
        self.time_tag += 1

        self.labels_ = labels
        self.cluster_centers_ = cluster_centers_indices

        return labels, cluster_centers_indices

    def __matrix_update(self, S, M, i, j):
        '''Matrix update for Incremental Affinity Propagation
        Args:
            S: similarity matrix
            M: matrix to update (Availability or Responsibility)
            i: row index
            j: column index
        '''

        # update as in the original paper
        old_N = M.shape[1]

        if i >= old_N:
            i = S[:old_N, i].argmax()
            return M[i, j]

        if j >= old_N:
            j = S[j, :old_N].argmax()
            return M[i, j]


class AffinityPropagationPosteriori:
    def __init__(self, trim:float=2, damping:float=0.7, max_iter:int=200):
        self.time_tag = 0
        self.data = list()
        self.trim = trim
        self.damping = damping
        self.max_iter = max_iter

    def _trim(self, X:np.array, labels:np.array, ignore=None) -> (np.array, np.array):
        '''Trim small size clusters'''

        unique, counts = np.unique(labels, return_counts=True)

        if ignore is None:
            condition = (counts > X.shape[0]*self.trim/100)
        else:
            condition = (counts > X.shape[0]*self.trim/100) | (np.isin(unique, ignore))

        survivors = unique[condition]

        X_trim = X[np.isin(labels, survivors)]
        labels_trim = labels[np.isin(labels, survivors)]

        return X_trim, labels_trim

    def _pack(self, X:np.array, labels:np.array) -> (np.array, np.array):
        '''Pack clusters of vectors into single representations'''
        unique = np.unique(labels)
        return np.array([X[labels==label].mean(0) for label in unique]), unique

    def fit(self, X: np.array) -> None:

        self.data.append(X)

        # standard AP
        if self.time_tag == 0:
            ap = AffinityPropagation(damping=self.damping, affinity='precomputed', max_iter=self.max_iter)
            ap.fit(pairwise_similarity(X))

            self._X_trim, self._labels_trim = self._trim(X, ap.labels_)
            self._X_pack, self._labels_pack = self._pack(self._X_trim, self._labels_trim)
            self.X_, self.labels_ = self._X_trim, self._labels_trim

            # didnt converge or single cluster
            if self.labels_.shape[0] <= 1:
                self.X_, self.labels_ = X, np.array([1] * X.shape[0])
                self._X_pack = np.array([X.mean(0)])
                self._labels_pack = np.array([1])
                self._X_trim, self._labels_trim = self.X_, self.labels_

        else:
            X_prev, X_curr = self.X_, X
            X_prev_pack = self._X_pack
            X_next = np.concatenate([X_prev_pack, X_curr])

            # preference over packed vectors
            preference = None####np.array([10.0]*len(X_prev_pack) + [0.1]*len(X_curr))
            ap = AffinityPropagation(damping=self.damping, affinity='precomputed', max_iter=self.max_iter, preference=preference)
            ap.fit(pairwise_similarity(X_next))

            labels_prev, labels_next = self.labels_, ap.labels_
            labels_prev_pack = self._labels_pack
            new_labels_prev_pack = labels_next[:labels_prev_pack.shape[0]]
            labels_curr = labels_next[labels_prev_pack.shape[0]:]

            # map old pack labels to new label
            new_labels_prev = labels_prev.copy()
            for old_, new_ in zip(labels_prev_pack, new_labels_prev_pack):
                new_labels_prev[labels_prev == old_] = new_

            # trim t1|t0
            self._prev_X_trim, self._prev_labels_trim = self._X_trim, new_labels_prev
            self._prev_X_pack, self._prev_labels_pack = self._X_pack, new_labels_prev_pack
            X_next = np.concatenate([self._prev_X_trim, X_curr])
            labels_next = np.concatenate([self._prev_labels_trim, labels_curr])

            X_next_trim, labels_next_trim = self._trim(X_next, labels_next)#, ignore=new_labels_prev)

            survivors = np.unique(labels_next_trim)

            self._prev_X_trim, self._prev_labels_trim = self._prev_X_trim[np.isin(self._prev_labels_trim, survivors)], self._prev_labels_trim[np.isin(self._prev_labels_trim, survivors)]

            self._curr_X_trim, self._curr_labels_trim = X_curr[np.isin(labels_curr, survivors)], labels_curr[np.isin(labels_curr, survivors)]
            self._curr_X_pack, self._curr_labels_pack = self._pack(self._curr_X_trim, self._curr_labels_trim)

            self.X_, self.labels_ = X_next_trim, labels_next_trim
            self._X_pack, self._labels_pack = self._pack(X_next_trim, labels_next_trim)

        self.time_tag += 1