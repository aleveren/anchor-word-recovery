from __future__ import division, print_function

import numpy as np
from numpy.random import RandomState

def findAnchors(Q, params, candidates):
    # Random number generator for generating dimension reduction
    prng_W = RandomState(params.seed)
    new_dim = params.new_dim

    # row normalize Q
    row_sums = Q.sum(axis=1)
    for i in range(Q.shape[0]):
        Q[i, :] /= float(row_sums[i])

    # Reduced dimension random projection method for recovering anchor words
    Q_red = Random_Projection(Q.T, new_dim, prng_W)
    Q_red = Q_red.T
    anchor_indices = Projection_Find(Q_red, params.K, candidates)

    # restore the original Q
    for i in range(Q.shape[0]):
        Q[i, :] *= float(row_sums[i])

    return anchor_indices

# Project the columns of the matrix M into the
# lower dimension new_dim
def Random_Projection(M, new_dim, prng):
    old_dim = M.shape[0]
    p = np.array([1./6, 2./3, 1./6])
    R = np.sqrt(3) * prng.choice([-1, 0, 1], p = p, size = (new_dim, old_dim))
    M_red = np.dot(R, M)
    return M_red

def Projection_Find(M, K, candidates):
    M = M.copy()
    n, dim = M.shape

    # stored recovered anchor words
    anchor_indices = np.zeros(K, dtype=np.int)

    # store the basis vectors of the subspace spanned by the anchor word vectors
    basis = np.zeros((K-1, dim))

    # find the farthest point p1 from the origin
    max_dist = 0
    for i in candidates:
        dist = np.dot(M[i], M[i])
        if dist > max_dist:
            max_dist = dist
            anchor_indices[0] = i

    # let p1 be the origin of our coordinate system, and
    # find the farthest point from p1
    first_anchor_word = M[anchor_indices[0]].copy()
    max_dist = 0
    for i in candidates:
        M[i] -= first_anchor_word
        dist = np.dot(M[i], M[i])
        if dist > max_dist:
            max_dist = dist
            anchor_indices[1] = i
            basis[0] = M[i] / np.linalg.norm(M[i])

    # stabilized gram-schmidt which finds new anchor words to expand our subspace
    for j in range(1, K - 1):
        # project all the points onto our basis and find the farthest point
        max_dist = 0
        for i in candidates:
            M[i] -= np.dot(M[i], basis[j-1])*basis[j-1]
            dist = np.dot(M[i], M[i])
            if dist > max_dist:
                max_dist = dist
                anchor_indices[j + 1] = i
                basis[j] = M[i] / np.linalg.norm(M[i])

    return list(anchor_indices)
