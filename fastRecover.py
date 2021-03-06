from __future__ import division, print_function

import time
import sys
import numpy as np
import multiprocessing

def do_recovery(Q, anchors, params):
    if params.loss == "originalRecover":
        return (Recover(Q, anchors), None)
    elif params.loss == "KL" or "L2" in params.loss:
        return nonNegativeRecover(Q, anchors, params)
    else:
        print("unrecognized loss function", loss, ". Options are KL,L2 or originalRecover")
        return None

def logsum_exp(y):
    m = y.max()
    return m + np.log((np.exp(y - m)).sum())

def KL_helper(arg):
    p,q = arg
    if p == 0:
        return 0
    return p*(np.log(p)-np.log(q))

def entropy(p):
    e = 0
    for i in range(len(p)):
        if p[i] > 0:
            e += p[i]*np.log(p[i])
    return -e

def KL(p,log_p,q):
    N = p.size
    ret = 0
    log_diff = log_p - np.log(q)
    ret = np.dot(p, log_diff)
    if ret < 0 or np.isnan(ret):
        print("invalid KL!")
        print("p:")
        for i in range(n):
            print(p[i])
            if p[i] <= 0:
                print("!!")
        print("\nq:")
        for i in range(n):
            print(q[i])
            if q[i] <= 0:
                print("!!")
        if ret < 0:
            print("less than 0", ret)
        sys.exit(1)
    return ret

#this method does not use a line search and as such may be faster
#but it needs an initialization of the stepsize
def fastQuadSolveExpGrad(y, x, eps, initialStepsize, anchorsTimesAnchors=None):
    (K,n) = x.shape

    # Multiply the target vector y and the anchors matrix X by X'
    #  (XX' could be passed in as a parameter)
    if anchorsTimesAnchors is None:
        print("XX' was not passed in")
        anchorsTimesAnchors = np.dot(x, x.transpose())
    targetTimesAnchors = np.dot(y, x.transpose())

    alpha = np.ones(K)/K
    log_alpha = np.log(alpha)

    iteration = 1
    eta = 0.1

    # To get the gradient, do one K-dimensional matrix-vector product
    proj = -2*(targetTimesAnchors - np.dot(alpha,anchorsTimesAnchors))
    new_obj = np.linalg.norm(proj,2)
    gap = float('inf')

    while True:
        # Set the learning rate
        eta = initialStepsize/(iteration**0.5)
        iteration += 1

        # Save previous values for convergence tests
        old_obj = new_obj
        old_alpha = np.copy(alpha)

        # Add the gradient and renormalize in logspace, then exponentiate
        log_alpha += -eta*proj
        log_alpha -= logsum_exp(log_alpha)

        alpha = np.exp(log_alpha)

        # Recalculate the gradient and check for convergence
        proj = -2*(targetTimesAnchors - np.dot(alpha,anchorsTimesAnchors))
        new_obj = np.linalg.norm(proj,2)

        # Stop if the L2 norm of the change in alpha OR 
        #  the % change in L2 norm of the gradient are below tolerance.
        #convergence = np.min(np.linalg.norm(alpha-old_alpha, 2), np.abs(new_obj-old_obj)/old_obj)

        # stop if the primal-dual gap < eps
        lam = np.copy(proj)
        lam -= lam.min()

        gap = np.dot(alpha, lam)

        if gap < eps and iteration > 1:
            break

        if iteration % 10000 == 0:
            print("iter", iteration, "obj", old_obj, "gap", gap)

    return alpha, iteration, new_obj, None, gap

def quadSolveExpGrad(y, x, eps, alpha=None, XX=None): 
    c1 = 10**(-4)
    c2 = 0.75
    if XX is None:
        print('making XXT')
        XX = np.dot(x, x.transpose())

    XY = np.dot(x, y)
    YY = float(np.dot(y, y))

    start_time = time.time()
    y_copy = np.copy(y)
    x_copy = np.copy(x)

    (K,n) = x.shape
    if alpha is None:
        alpha = np.ones(K)/K

    old_alpha = np.copy(alpha)
    log_alpha = np.log(alpha)
    old_log_alpha = np.copy(log_alpha)

    it = 1 
    aXX = np.dot(alpha, XX)
    aXY = float(np.dot(alpha, XY))
    aXXa = float(np.dot(aXX, alpha.transpose()))

    grad = 2*(aXX-XY)
    new_obj = aXXa - 2*aXY + YY

    old_grad = np.copy(grad)

    stepsize = 1
    repeat = False
    decreased = False
    gap = float('inf')
    while 1:
        eta = stepsize
        old_obj = new_obj
        old_alpha = np.copy(alpha)
        old_log_alpha = np.copy(log_alpha)
        if new_obj == 0:
            break
        if stepsize == 0:
            break

        it += 1
        #if it % 1000 == 0:
        #    print("\titer", it, new_obj, gap, stepsize)
        #update
        log_alpha -= eta*grad
        #normalize
        log_alpha -= logsum_exp(log_alpha)
        #compute new objective
        alpha = np.exp(log_alpha)

        aXX = np.dot(alpha, XX)
        aXY = float(np.dot(alpha, XY))
        aXXa = float(np.dot(aXX, alpha.transpose()))

        old_obj = new_obj
        new_obj = aXXa - 2*aXY + YY
        if not new_obj <= old_obj + c1*stepsize*np.dot(grad, alpha - old_alpha): #sufficient decrease
            stepsize /= 2.0 #reduce stepsize
            alpha = old_alpha 
            log_alpha = old_log_alpha
            new_obj = old_obj
            repeat = True
            decreased = True
            continue

        #compute the new gradient
        old_grad = np.copy(grad)
        grad = 2*(aXX-XY)
        
        if (not np.dot(grad, alpha - old_alpha) >= c2*np.dot(old_grad, alpha-old_alpha)) and (not decreased): #curvature
            stepsize *= 2.0 #increase stepsize
            alpha = old_alpha
            log_alpha = old_log_alpha
            grad = old_grad
            new_obj = old_obj
            repeat = True
            continue

        decreased = False

        lam = np.copy(grad)
        lam -= lam.min()
        
        gap = np.dot(alpha, lam)
        convergence = gap
        if (convergence < eps):
            break

    return alpha, it, new_obj, stepsize, gap

def KLSolveExpGrad(y,x,eps, alpha=None):
    s_t = time.time()
    c1 = 10**(-4)
    c2 = 0.9
    it = 1 

    start_time = time.time()
    y = clip(y, 0, 1)
    x = clip(x, 0, 1)

    (K,N) = x.shape
    mask = list(nonzero(y)[0])

    y = y[mask]
    x = x[:, mask]

    x += 10**(-9)
    x /= x.sum(axis=1)[:,newaxis]

    if alpha is None:
        alpha = np.ones(K)/K

    old_alpha = np.copy(alpha)
    log_alpha = np.log(alpha)
    old_log_alpha = np.copy(log_alpha)
    proj = np.dot(alpha,x)
    old_proj = np.copy(proj)

    log_y = np.log(y)
    new_obj = KL(y,log_y, proj)
    y_over_proj = y/proj
    grad = -np.dot(x, y_over_proj.transpose())
    old_grad = np.copy(grad)

    stepsize = 1
    decreasing = False
    repeat = False
    gap = float('inf')

    while 1:
        eta = stepsize
        old_obj = new_obj
        old_alpha = np.copy(alpha)
        old_log_alpha = np.copy(log_alpha)

        old_proj = np.copy(proj)

        it += 1
        #take a step
        log_alpha -= eta*grad

        #normalize
        log_alpha -= logsum_exp(log_alpha)

        #compute new objective
        alpha = np.exp(log_alpha)
        proj = np.dot(alpha,x)
        new_obj = KL(y,log_y,proj)
        if new_obj < eps:
            break

        grad_dot_deltaAlpha = np.dot(grad, alpha - old_alpha)
        assert (grad_dot_deltaAlpha <= 10**(-9))
        if not new_obj <= old_obj + c1*stepsize*grad_dot_deltaAlpha: #sufficient decrease
            stepsize /= 2.0 #reduce stepsize
            if stepsize < 10**(-6):
                break
            alpha = old_alpha 
            log_alpha = old_log_alpha
            proj = old_proj
            new_obj = old_obj
            repeat = True
            decreasing = True
            continue

        #compute the new gradient
        old_grad = np.copy(grad)
        y_over_proj = y/proj
        grad = -np.dot(x, y_over_proj)

        if not np.dot(grad, alpha - old_alpha) >= c2*grad_dot_deltaAlpha and not decreasing: #curvature
            stepsize *= 2.0 #increase stepsize
            alpha = old_alpha
            log_alpha = old_log_alpha
            grad = old_grad
            proj = old_proj
            new_obj = old_obj
            repeat = True
            continue

        decreasing= False
        lam = np.copy(grad)
        lam -= lam.min()
        
        gap = np.dot(alpha, lam)
        convergence = gap
        if (convergence < eps):
            break

    return alpha, it, new_obj, stepsize, time.time()- start_time, gap

def Recover(Q, anchors):
    K = len(anchors)
    orig = Q
    permutation = range(len(Q[:,0]))
    for a in anchors:
        permutation.remove(a)
    permutation = anchors + permutation
    Q_prime = Q[permutation, :]
    Q_prime = Q_prime[:, permutation]
    DRD = Q_prime[0:K, 0:K]
    DRAT = Q_prime[0:K, :]
    DR1 = np.dot(DRAT, np.ones(DRAT[0,:].size))
    z = np.linalg.solve(DRD, DR1)
    A = np.dot(np.linalg.inv(np.dot(DRD, np.diag(z))), DRAT).transpose()
    reverse_permutation = [0]*(len(permutation))
    for p in permutation:
        reverse_permutation[p] = permutation.index(p)
    A = A[reverse_permutation, :]
    return A

def fastRecover(args):
    y, x, v, anchors, divergence, XXT, initial_stepsize, epsilon = args
    start_time = time.time() 

    K = len(anchors)
    alpha = np.zeros(K)
    gap = None

    if v in anchors:
        alpha[anchors.index(v)] = 1
        it = -1
        dist = 0
        stepsize = 0

    else:
        if divergence == "KL":
            alpha, it, dist, stepsize, t, gap = KLSolveExpGrad(y=y, x=x, eps=epsilon)
        elif divergence == "L2":
            alpha, it, dist, stepsize, gap = quadSolveExpGrad(y=y, x=x, eps=epsilon, alpha=None, XX=XXT)
        elif divergence == "fastL2":
            alpha, it, dist, stepsize, gap = fastQuadSolveExpGrad(y=y, x=x, eps=epsilon, initialStepsize=100, anchorsTimesAnchors=XXT)
        else:
            raise ValueError("Invalid divergence!")

        if np.isnan(alpha).any():
            alpha = np.ones(K) / K

    end_time = time.time()
    return (v, it, dist, alpha, stepsize, end_time - start_time, gap)

class myIterator:
    def __init__(self, Q, anchors, divergence, v_max, initial_stepsize, epsilon=10**(-7)):
        self.Q = Q
        self.anchors = anchors
        self.v = -1
        self.V_max = v_max
        self.divergence = divergence
        self.X = self.Q[anchors, :]
        if "L2" in divergence:
            self.anchorsTimesAnchors = np.dot(self.X, self.X.transpose())
        else:
            self.anchorsTimesAnchors = None
        self.initial_stepsize = initial_stepsize
        self.epsilon = epsilon

    def __iter__(self):
        return self
    def next(self):
        return self.__next__()
    def __next__(self):
        self.v += 1
       # print("generating word", self.v, "of", self.V_max)
        if self.v >= self.V_max:
            raise StopIteration
            return 0
        v = self.v
        Q = self.Q
        anchors = self.anchors
        divergence = self.divergence
        return (np.copy(Q[v, :]), np.copy(self.X), v, anchors, divergence, self.anchorsTimesAnchors, self.initial_stepsize, self.epsilon)

def nonNegativeRecover(Q, anchors, params, initial_stepsize=1):
    if params.outfile is not None:
        topic_likelihoodLog = open(params.outfile+".topic_likelihoods", 'w')
        word_likelihoodLog = open(params.outfile+".word_likelihoods", 'w')
        alphaLog = open(params.outfile+".alpha", 'w')

    V = Q.shape[0]
    K = len(anchors)

    P_w = np.diag(np.dot(Q, np.ones(V)))
    for v in range(V):
        if np.isnan(P_w[v,v]):
            P_w[v,v] = 10**(-16)

    #normalize the rows of Q_prime
    Q /= Q.sum(axis=1, keepdims=True)

    s = time.time()
    A = np.array(np.zeros((V, K)))
    if params.max_threads > 0:
        pool = multiprocessing.Pool(params.max_threads)
        print("begin threaded recovery with", params.max_threads, "processors")
        args = myIterator(Q, anchors, params.loss, V, initial_stepsize, params.eps)
        rows = pool.imap_unordered(fastRecover, args, chunksize = 10)
        for r in rows:
            v, it, obj, alpha, stepsize, t, gap = r
            A[v, :] = alpha
            if v % 1000 == 0:
                print("\t".join([str(x) for x in [v, it, max(alpha)]]))
                if params.outfile is not None:
                    print(v, alpha, file=alphaLog)
                    alphaLog.flush()
                sys.stdout.flush()

    else:
        X = Q[anchors, :]
        XXT = np.dot(X, X.transpose())
        for w in range(V):
            y = Q[w, :]
            v, it, obj, alpha, stepsize, t, gap = fastRecover((y, X, w, anchors, params.loss, XXT, initial_stepsize, params.eps))
            A[w, :] = alpha
            if v % 1 == 0:
                print("word", v, it, "iterations. Gap", gap, "obj", obj, "final stepsize was", stepsize, "took", t, "seconds")
                if params.outfile is not None:
                    print(v, alpha, file=alphaLog)
                    alphaLog.flush()
                sys.stdout.flush()

    #rescale A matrix
    #Bayes rule says P(w|z) proportional to P(z|w)P(w)
    A = np.dot(P_w, A)

    #normalize columns of A. This is the normalization constant P(z)
    colsums = A.sum(axis=0)

    for k in range(K):
        A[:, k] = A[:, k] / A[:,k].sum()

    if params.outfile is not None:
        for k in range(K):
            print(colsums[k], file=topic_likelihoodLog)

        for v in range(V):
            print(P_w[v,v], file=word_likelihoodLog)

        topic_likelihoodLog.close()
        word_likelihoodLog.close()
        alphaLog.close()

    return A, colsums
