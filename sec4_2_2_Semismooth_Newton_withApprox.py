## This script incorporates the Eigenvalue Tracking for approximation in eigenvalue
## decomposition for last iteration
## Warning : this method does NOT allow the algorithm to convergence
## Results are displayed in TRACKING_PLOT_LAST_ITER.py
##   Update of the algorithm presented by Yancheng Yuan to solve the dual of 
##   the following nearest correlation matrix problem : 
##   min 0.5 || X - M ||^2 in the Frobenius norm, such that
##   X_ii = b_i and X >= tau * I (generalized constraint, classic problem with b_i = 1 and tau = 0)
##   The method comes from " A Quadratically Convergent Newton Method for Computing 
##   the Nearest Correlation Matrix Porblem" by Houduo Qi and Defeng Sun (2006)

import numpy as np
import time
import matplotlib.pyplot as plt 
#from scipy.sparse.linalg import eigsh
 # Uses sparse diagonal matrix

# cg(A, b, x0=None, *, rtol=1e-05, atol=0.0, maxiter=None, M=None, callback=None)
# minres(A, b, x0=None, *, rtol=1e-05, shift=0.0, maxiter=None, M=None, callback=None, show=False, check=False)
# bicgstab(A, b, x0=None, *, rtol=1e-05, atol=0.0, maxiter=None, M=None, callback=None)
# bicg(A, b, x0=None, *, rtol=1e-05, atol=0.0, maxiter=None, M=None, callback=None)
# cgs(A, b, x0=None, *, rtol=1e-05, atol=0.0, maxiter=None, M=None, callback=None)
# gmres(A, b, x0=None, *, rtol=1e-05, atol=0.0, restart=None, maxiter=None, M=None, callback=None, callback_type=None)[source]

#Generate F(y) Compute the gradient
def dual_obj_grad(y_input, lamb, p_input, b_0, n):
    """
    Parameters
    ----------
    y_input : array of n dual variables 
        
    lamb : array of the n eigenvalues (only positive eigenvalues needed)
       
    p_input : matrix of n eigenvectors (only eigenvectors associated to positive eigenvectors needed)
    
    b_0 : right-side of the linear constraint (b_0 = e for correlation matrices)µ
    
    n : size of the input matrix of the initial problem

    Returns
    -------
    f : dual objective function 
    -y^T b + 0.5 ||( M + Diag(y))_+||_F^2   = -y^T b + 0.5 ||P \Lambda_+ P^T||_F^2 
                                            = -y^T b - 0.5 || \Lambda_+||_F^2
        
    Fy : diag((Diag(y) + M)_+) = Diag(P \Lambda_+ P^T) = sum( Pij^2 max(\lamb_j, 0))

    """
    # Initialization and Allocation
    f = 0.0
    Fy = np.zeros((n, 1))
    # P = P^T (in-place allocation)
    p_input_copy = (p_input.copy()).transpose()
    
    for i in range(0,n):
        p_input_copy[i, :] = ((np.maximum(lamb[i],0).astype(float))**0.5)*p_input_copy[i, :]

    for i in range(0,n):
        Fy[i] = np.sum(p_input_copy[:, i]*p_input_copy[:, i])
        f = f + np.square((np.maximum(lamb[i],0)))
        #since the norm is unitary invariant, ||P Lambda_+ P^T||^2_f = ||Lambda_+||_F^2

    f = 0.5*f - np.dot(b_0.transpose(), y_input) # - y^t b

    return f, Fy  


# use PCA to generate a primal feasible solution 

def my_pca(x_input, lamb, p_input, b_0, n):
    
    x_pca = x_input
    
    ### EIGENVALUES FILTRATION ###
    lamb = np.asarray(lamb)
    # select the positive eigenvalues 
    lp = lamb > 1e-6   
    # the rank is the number of positive eigenvalues 
    r = lamb[lp].size
    if r == 0: #take a zero matrix on case we have no positive eigenvalues 
        x_pca = np.zeros((n, n))
    elif r == n: #keep the original matrix if all eigenvalues are positive
        x_pca = x_input
    elif r<(n/2.0):  #construct a matrix based on the r gratest eigenvalues 
        lamb1 = lamb[lp].copy()
        lamb1 = lamb1.transpose()
        lamb1 = np.sqrt(lamb1.astype(float))
        P1 = p_input[:, 0:r].copy()
        if r>1:
            P1 = np.dot(P1,np.diagflat(lamb1))
            x_pca = np.dot(P1, P1.transpose())
        else:
            x_pca = np.square(lamb1) * np.dot(P1, P1.transpose())

    else:   #handle negative eigenvalues to ensure regularization 
        lamb2 = -lamb[r:n].copy()
        lamb2 = np.sqrt(lamb2.astype(float))
        p_2 = p_input[:, r:n]
        p_2 = np.dot(p_2,np.diagflat(lamb2))
        x_pca = x_pca + np.dot(p_2, p_2.transpose())

    # To make x_pca positive semidefinite with diagonal elements exactly b0
    d = np.diag(x_pca)
    d = d.reshape((d.size, 1))
    d = np.maximum(d, b_0.reshape(d.shape))
    x_pca = x_pca - np.diagflat(x_pca.diagonal()) + np.diagflat(d) #make diagonal equal to b0
    d = d.astype(float)**(-0.5)
    d = d*((np.sqrt(b_0.astype(float))).reshape(d.shape))
    x_pca = x_pca*(np.dot(d, d.reshape(1, d.size)))

    return x_pca
# end of PCA

#To generate the first order difference of lambda
# To generate the first order essential part of d


def my_omega_mat(p_input, lamb, n):
    idx_idp = np.where(lamb > 0)
    idx_idp = idx_idp[0]
    n = lamb.size
    r = idx_idp.size
    if r > 0:
        if r == n:
            omega_12 = np.ones((n, n))
        else:
            s = n - r
            dp = lamb[0:r].copy()
            dp = dp.reshape(dp.size, 1)
            dn = lamb[r:n].copy()
            dn = dn.reshape((dn.size, 1))
            omega_12 = np.dot(dp, np.ones((1, s)))
            omega_12 = omega_12/(np.dot(np.abs(dp), np.ones((1,s))) + np.dot(np.ones((r,1)), abs(dn.transpose())))
            omega_12 = omega_12.reshape((r, s))

    else:
        omega_12 = np.array([])

    return omega_12

# End of my_omega_mat


# To generate Jacobian


def my_jacobian_matrix(x, omega_12, p_input, n):
    #allocation
    x_result = np.zeros((n,1))
    [r, s] = omega_12.shape
    
    if r > 0:
        hmat_1 = p_input[:, 0:r].copy()
        if r < n/2.0:
            i=0
            while i<n:
                hmat_1[i,:] = x[i]*hmat_1[i,:]
                i = i+1

            omega_12 = omega_12*(np.dot(hmat_1.transpose(), p_input[:, r:n]))
            hmat = np.dot(hmat_1.transpose(),np.dot(p_input[:, 0:r], p_input[:, 0:r].transpose()))
            hmat = hmat + np.dot(omega_12, p_input[:, r:n].transpose())
            hmat = np.vstack((hmat, np.dot(omega_12.transpose(), p_input[:, 0:r].transpose())))
            i = 0
            while i<n:
                x_result[i] = np.dot(p_input[i, :], hmat[:, i])
                x_result[i] = x_result[i] + 1.0e-10*x[i]
                i = i+1

        else:
            if r==n:
                x_result = 1.0e-10*x
            else:
                hmat_2 = p_input[:, r:n].copy()
                i=0
                while i<n:
                    hmat_2[i, :] = x[i]*hmat_2[i, :]
                    i = i+1

                omega_12 = np.ones((r, s)) - omega_12
                omega_12 = omega_12*(np.dot(p_input[:, 0:r].transpose(), hmat_2))
                hmat = np.dot(p_input[:, r:n].transpose(), hmat_2)
                hmat = np.dot(hmat, p_input[:, r:n].transpose())
                hmat = hmat + np.dot(omega_12.transpose(),p_input[:, 0:r].transpose())
                hmat = np.vstack((np.dot(omega_12, p_input[:, r:n].transpose()), hmat))
                i=0
                while i<n:
                    x_result[i] = np.dot(-p_input[i,:], hmat[:, i])
                    x_result[i] = x[i] + x_result[i] + 1.0e-10*x[i]
                    i = i+1

    return x_result

#end of Jacobian

# PCG Method in case of use of preconditioned conjugate gradient method


def my_pre_cg(b, tol, maxit, c, omega_12, p_input, n):
    #Initializations
    r = b.copy()
    r = r.reshape(r.size, 1)
    c = c.reshape(c.size, 1)
    n2b = np.linalg.norm(b)
    tolb = tol*n2b
    p = np.zeros((n, 1))
    flag = 1
    iterk = 0
    relres = 1000
    # Precondition
    z = r/c
    rz_1 = np.dot(r.transpose(), z)
    rz_2 = 1
    d = z.copy()
    # CG Iteration
    for k in range(0, maxit):
        if k > 0:
            beta = rz_1/rz_2
            d = z + beta*d

        w = my_jacobian_matrix(d, omega_12, p_input, n)
        denom = np.dot(d.transpose(), w)
        iterk = k+1
        relres = np.linalg.norm(r)/n2b
        if denom <= 0:
            p = d/np.linalg.norm(d)
            break
        else:
            alpha = rz_1/denom
            p = p + alpha*d
            r = r - alpha*w

        z = r/c
        if np.linalg.norm(r)<=tolb: #exit if hmat p = b solved in relative error tolerance
            iterk = k+1
            relres = np.linalg.norm(r)/n2b
            flag = 0
            break

        rz_2 = rz_1
        rz_1 = np.dot(r.transpose(), z)

    return p, flag, relres, iterk

# end of pre_cg

# start of minres in case of use of preconditioned minimimization of residues method
def my_minres(b, tol, maxit, c, omega_12, p_input, n):
    # Initializations
    n2b = np.linalg.norm(b)
    r = b.copy()
    r = r.reshape(r.size, 1)
    c = c.reshape(c.size, 1)
    p = np.zeros((n, 1))
    n2b = np.linalg.norm(b)
    tolb = tol * n2b
    z = r/c
    d0 = z
    d1 = z
    w0 = my_jacobian_matrix(z, omega_12, p_input, n)
    w1 = w0
    flag = 1
    iterk = 0
    relres = 1000
    
    #MINRES iteration
    for k in range(0, maxit):
        d2 = d1; d1 = d0
        w2 = w1; w1 = w0
        alpha = np.dot(r.transpose(),w1)/ np.dot(w1.transpose(), w1)
        p = p + alpha * d1
        r = r - alpha * w1

        if np.linalg.norm(r)<=tolb: 
            iterk = k+1
            relres = np.linalg.norm(r)/n2b
            flag = 0
            break
        z = r/c
        p0 = z 
        w0 = my_jacobian_matrix(p0, omega_12, p_input, n)
        
        Beta1 = np.dot(w0.transpose(), w1)/np.dot(w1.transpose(), w1)
        
        d0 = d0 - Beta1 * d1
        w0 = w0 - Beta1 * w1
        if iterk > 1 : 
            Beta2 =  np.dot(w0.transpose(), w2)/np.dot(w2.transpose(), w2)
            d0 = d0 - Beta2 * d2
            w0 = w0 - Beta2 * w2
       
        
    
    return p, flag, relres, iterk

# end of minres 

#to generate the diagonal preconditioner
def my_precond_matrix(omega_12, p_input, n):
    [r, s] = omega_12.shape
    c = np.ones((n, 1)) #allocation
    if r > 0:
        if r < n/2.0:
            hmat = (p_input.copy()).transpose()
            hmat = hmat*hmat
            hmat_12 = np.dot(hmat[0:r, :].transpose(), omega_12)
            d = np.ones((r, 1))
            for i in range(0, n):
                c_temp = np.dot(d.transpose(), hmat[0:r, i])
                c_temp = c_temp*hmat[0:r, i]
                c[i] = np.sum(c_temp)
                c[i] = c[i] + 2.0*np.dot(hmat_12[i, :], hmat[r:n, i])
                if c[i] < 1.0e-8:
                    c[i] = 1.0e-8

        else:
            if r < n:
                hmat = (p_input.copy()).transpose()
                hmat = hmat*hmat
                omega_12 = np.ones((r,s)) - omega_12
                hmat_12 = np.dot(omega_12, hmat[r:n, :])
                d = np.ones((s, 1))
                dd = np.ones((n, 1))

                for i in range(0, n):
                    c_temp = np.dot(d.transpose(), hmat[r:n, i])
                    c[i] = np.sum(c_temp*hmat[r:n, i])
                    c[i] = c[i] + 2.0*np.dot(hmat[0:r, i].transpose(), hmat_12[:, i])
                    alpha = np.sum(hmat[:, i])
                    c[i] = alpha*np.dot(hmat[:, i].transpose(), dd) - c[i]
                    if c[i] < 1.0e-8:
                        c[i] = 1.0e-8

    return c

# end of precond_matrix 


# my_issorted()
def my_issorted(x_input, flag):
    """
    Check if the x_input is in the order described by flag 
    if flag = 1, it checks if x_input is in the increasing order 
    if flag = -1, same in the decreasing order 

    """
    n = x_input.size
    tf_value = False
    if n < 2:
        tf_value = True
    else:
        if flag == 1:
            i = 0
            while i < n-1:
                if x_input[i] <= x_input[i+1]:
                    i = i+1
                else:
                    break

            if i == n-1:
                tf_value = True
            elif i < n-1:
                tf_value = False

        elif flag == -1:
            i = n-1
            while i > 0:
                if x_input[i] <= x_input[i-1]:
                    i = i-1
                else:
                    break

            if i == 0:
                tf_value = True
            elif i > 0:
                tf_value = False

    return tf_value
# end of my_issorted()
def plot_eigenvalues_histogram(eigenvalues, bins=50, log_scale=True):
    """
    Plots a histogram of eigenvalues.

    Parameters:
        eigenvalues (array-like): Array of eigenvalues to be plotted.
        bins (int): Number of bins for the histogram.
        log_scale (bool): If True, set the y-axis to a logarithmic scale.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(eigenvalues, bins=bins, log=log_scale, edgecolor='k', alpha=0.75)
    plt.xlabel("Eigenvalues")
    plt.ylabel("Frequency (log scale)" if log_scale else "Frequency")
    plt.title("Histogram of Eigenvalues")
    plt.grid(True)
    plt.show()
  
def first_order_perturbation_vectorized(lam, U, delty):
    """
    Compute first-order perturbation corrections to the eigenvalues and eigenvectors
    of a Hermitian matrix M perturbed by Diag(y).

    Parameters:
      lam : np.ndarray
          Array of initial eigenvalues, sorted in descending order.
      U : np.ndarray
          Matrix of initial eigenvectors.
      delty : np.ndarray
          Perturbation vector of size (n,), used in Diag(y).
          
    Returns:
      lam_pert : np.ndarray
          Approximated eigenvalues of M + Diag(y).
      U_pert : np.ndarray
          Matrix whose columns are the approximated eigenvectors.
    """
    # Compute correction matrix A = U.T @ Diag(y) @ U efficiently
    delty = delty.flatten()
    assert U.ndim == 2, f"U has {U.ndim} dimensions, expected 2"
    assert delty.ndim == 1, f"delty has {delty.ndim} dimensions, expected 1"
    assert U.shape[0] == delty.shape[0], f"Incompatible shapes: U {U.shape}, delty {delty.shape}"
    U = np.asarray(U)
    print(U.shape)
    print(delty[:,None].shape)
    A = delty[:,None] * U
    print("Element-wise product finished")
    A = np.dot(U.T, A)
    print("Computation of A finished")
    # Compute weighting matrix 
    lam = lam.flatten()
    diffs = lam[:, None] - lam[None, :]  # (n, n) matrix    
    np.fill_diagonal(diffs, np.inf)  # avoid division by zero on the diagonal 
    # --- Correction to the eigenvalues ---
    lam = lam + np.diag(A)
    # --- Correction to the eigenvectors --- 
    A = A.transpose() / diffs
    # Compute corrections to the eigenvectors
    U = U + np.dot(U, A)
    print("Correction to eigenvectors finished")
    # Normalize each eigenvector (each column)
    U = U / np.linalg.norm(U, axis=0, keepdims=True)
    return lam, U

def my_mexeig(x_input, deltay, lam, U):
    """
    Computation of the eigenvalues of x_input, with eigenvalues in a decreasing order 
    """
    [n, m] = x_input.shape
    
    if (np.abs(np.mean(deltay)) < 0.5 ) : # and np.var(deltay) < 1) : 
        print("first order approximation of eigenvalues")
        [lamb, p_x] = first_order_perturbation_vectorized(lam, U, deltay)
    else : 
        print("exact computation of eigenvalues")
        [lamb, p_x] = np.linalg.eigh(x_input)
    
    p_x = p_x.real
    lamb = lamb.real
    lamb[np.abs(lamb) < 1e-6] = 0
    # we want eigenvalues in a decreasing order 
    if my_issorted(lamb, 1): #if in increasing order, it suffices to invert the order
    
        lamb = lamb[::-1]
        p_x = np.fliplr(p_x)
        
    elif my_issorted(lamb, -1): #in this case, it's already done 
        return p_x, lamb
    else: #otherwise, we need to classify the eigenvalues and corresponding eigenvectors
        idx = np.argsort(-lamb)

        lamb = lamb[idx]

        p_x = p_x[:, idx]

    #lamb = lamb.reshape((n, 1))
    #p_x = p_x.reshape((n, n))
    return p_x, lamb

# end of my_mymexeig()


# begin of the main function
def my_correlationmatrix(M, b_input=None, tau=None, tol=None):
    ### Allocation of tables to stock information about the method
    
    DualObjValue = []
    PrimalObjValue = []
    RelativeDualityGap = []
    GradNorm = []
    exectime = dict()
    inneriter = []
    #y_tracking = dict()
    rank_track = []
    rank_positive_track = []
    ####
    
    print ('-- Semismooth Newton-CG method starts -- \n')
    
    [n, m] = M.shape #in principle, n = m for our case
    
    # work on a copy to not affect the data
    M = M.copy()
    
    # clock start for init step
    t0 = time.perf_counter()
    # ensure symmetry (generalization of bosdorf for nonnecessary square input matrices, not useful in our case)
    #M = (M + M.transpose())/2.0
    # A(X) = b (in classic problem, diag(X) = e = np.ones((n,1)))
    b_g = np.ones((n, 1))
    # error tolerance by default 
    error_tol = 1.0e-6
    
    ###
    if b_input is None: 
        tau = 0
    elif tau is None: # only requirement of symmetric positive semidefiniteness
        b_g = b_input.copy()
        tau = 0
    elif tol is None: 
        b_g = b_input.copy() - tau*np.ones((n, 1))
        M = M - tau*np.eye(n, n)
    else:
        b_g = b_input.copy() - tau*np.ones((n, 1))
        M = M - tau*np.eye(n, n)
        error_tol = np.maximum(1.0e-12, tol)
    ###
    
    res_b = np.zeros((300,1))
    
    norm_b0 = np.linalg.norm(b_g) 
    
    # allocation 
    y = np.zeros((n, 1))   # n dual variables 
    f_y = np.zeros((n, 1)) # n values for the gradient   
    deltay = np.ones((n, 1)) * 1000
    lamb = np.zeros((n, 1))
    p_x = np.zeros((n, n))
    # init
    k=0         
    f_eval = 0       # init of the number of function evaluations
    iter_whole = 13  # maximum number of the outer algo 
    iter_inner = 20  # maximum number of line search in Newton method
    maxit = 200      # maximum number of iterations in PCG
    iterk = 0
    #inner = 0
    tol_cg = 1.0e-2  # relative accuracy for CGs
    sigma_1 = 1.0e-4 
    
    x0 = y.copy()
    prec_time = 0
    pcg_time = 0
    eig_time = 0
    c = np.ones((n, 1))
    d = np.zeros((n, 1))
    # val_g = 0.5 || M ||_F^2 (constant term in the dual objective)
    val_g = 0.5 * np.sum((M.astype(float))*(M.astype(float)))
    
    
    ###########################################################################
    x_result = M + np.diagflat(y) # = M 
    # ensure symmetry
    x_result = (x_result + x_result.transpose())/2.0
    # time for the eigenvalue decomposition of x_result 
    eig_time0 = time.perf_counter()
    [p_x, lamb] = my_mexeig(x_result, deltay, lamb, p_x)  
    #plot_eigenvalues_histogram(lamb)
    rank_track.append(len(lamb[lamb != 0]))
    rank_positive_track.append(len(lamb[lamb > 0]))
    eig_time = eig_time + (time.perf_counter() - eig_time0)
    # compute objective of dual and F_y
    [f_0, f_y] = dual_obj_grad(y, lamb, p_x, b_g, n) 
   
    ###########################################################################
    # value of the dual objective function 
    initial_f = val_g - f_0   
    #PCA provide a good initial point for the investigation since it provides 
    #a feasible solution 
    x_result = my_pca(x_result, lamb, p_x, b_g, n)
    # value of the primal objective function
    val_obj = np.sum(((x_result - M)*(x_result - M)))/2.0 

    gap = (val_obj - initial_f)/(1.0 + np.abs(initial_f) + np.abs(val_obj))
    f = f_0.copy()
    
    f_eval = f_eval + 1
    
    #gradient of the dual objective -b + diag([diag(y)+M]_+)
    b_input = b_g - f_y
    #norm of the gradient of the dual objective function
    norm_b = np.linalg.norm(b_input)
    
    #clock end for init step
    time_used = time.perf_counter() - t0
    
    print ('Newton-CG: Initial Dual objective function value: %s \n' % initial_f)
    DualObjValue.append(initial_f)
    print ('Newton-CG: Initial Primal objective function value: %s \n' % val_obj)
    PrimalObjValue.append(val_obj)
    print ('Newton-CG: Norm of Gradient: %s \n' % norm_b)
    GradNorm.append(norm_b)
    print ('Newton-CG: computing time used so far: %s \n' % time_used)
    exectime['init'] = time_used
    
    omega_12 = my_omega_mat(p_x, lamb, n)
    x0 = y.copy()
    #y_tracking[0] = x0
    while np.abs(gap) > error_tol and norm_b/(1+norm_b0) > error_tol and k < iter_whole:
        #preconditioning 
        prec_time0 = time.perf_counter()
        c = my_precond_matrix(omega_12, p_x, n)
        prec_time = prec_time + (time.perf_counter() - prec_time0)
    
        pcg_time0 = time.perf_counter()
        [d, flag, relres, iterk] = my_pre_cg(b_input, tol_cg, maxit, c, omega_12, p_x, n)
        #[d, flag, relres, iterk] = my_minres(b_input, tol_cg, maxit, c, omega_12, p_x, n)
        pcg_time = pcg_time + (time.perf_counter() - pcg_time0)
        inneriter.append(iterk)
        print ('Newton-CG: Number of CG Iterations=== %s \n' % iterk)
        
        if flag == 1:
            print ('=== Not a completed Newton-CG step === \n')

        slope = np.dot((f_y - b_g).transpose(), d)
        deltay = (x0 + d) - y
        y = (x0 + d).copy()
        #y_tracking[k+1]=y
        #######################################################################
        x_result = M + np.diagflat(y)
        x_result = (x_result + x_result.transpose())/2.0
        eig_time0 = time.perf_counter()
        [p_x, lamb] = my_mexeig(x_result,deltay, lamb, p_x)
        #plot_eigenvalues_histogram(lamb)
        rank_track.append(len(lamb[lamb != 0]))
        rank_positive_track.append(len(lamb[lamb > 0]))
        eig_time = eig_time + (time.perf_counter() - eig_time0)
        [f, f_y] = dual_obj_grad(y, lamb, p_x, b_g, n)
        #######################################################################
        k_inner = 0
        #inner loop, armijo condition 
        while k_inner <= iter_inner and f > f_0 + sigma_1*(np.power(0.5, k_inner))*slope + 1.0e-6:
            k_inner = k_inner + 1
            deltay = x0 + (np.power(0.5, k_inner))*d - y 
            y = x0 + (np.power(0.5, k_inner))*d
            ###################################################################
            x_result = M + np.diagflat(y)
            x_result = (x_result + x_result.transpose())/2.0
            eig_time0 = time.perf_counter()
            [p_x, lamb] = my_mexeig(x_result,deltay, lamb, p_x)
            #[p_x, lamb] = my_mexeig_iterative(x_result, lamb)
            rank_track.append(len(lamb[lamb != 0]))
            rank_positive_track.append(len(lamb[lamb > 0]))
            eig_time = eig_time + (time.perf_counter() - eig_time0)
            [f, f_y] = dual_obj_grad(y, lamb, p_x, b_g, n)
            ###################################################################
        f_eval = f_eval + k_inner + 1
        x0 = y.copy()
        f_0 = f.copy()
        
        val_dual = val_g - f_0
        x_result = my_pca(x_result, lamb, p_x, b_g, n)
        val_obj = np.sum((x_result - M)*(x_result - M))/2.0
        gap = (val_obj - val_dual)/(1 + np.abs(val_dual) + np.abs(val_obj))
        print ('Newton-CG: The relative duality gap: %s \n' % gap)
        RelativeDualityGap.append(gap)
        print ('Newton-CG: The Dual objective function value: %s \n' % val_dual)
        DualObjValue.append(val_dual)
        print ('Newton-CG: The Primal objective function value: %s \n' % val_obj)
        PrimalObjValue.append(val_obj)

        b_input = b_g - f_y
        norm_b = np.linalg.norm(b_input)
        time_used = time.perf_counter() - t0
        rel_norm_b = norm_b/(1+norm_b0)
        print ('Newton-CG: Norm of Gradient: %s \n' % norm_b)
        GradNorm.append(norm_b)
        print ('Newton-CG: Norm of Relative Gradient: %s \n' %  rel_norm_b)
        print ('Newton-CG: Computing time used so for %s \n' % time_used)
        exectime['step'] = time_used
        res_b[k] = norm_b
        k = k + 1
        omega_12 = my_omega_mat(p_x, lamb, n)

    position_rank = np.maximum(lamb, 0)>0
    rank_x = (np.maximum(lamb, 0)[position_rank]).size
    final_f = val_g - f
    x_result = x_result + tau*(np.eye(n))
    time_used = time.perf_counter() - t0
    print ('\n')

    print ('Newton-CG: Number of iterations: %s \n' % k)
    print ('Newton-CG: Number of Funtion Evaluation:  =========== %s\n' % f_eval)
    print ('Newton-CG: Final Dual Objective Function value: ========= %s\n' % final_f)
    DualObjValue.append(final_f)
    print ('Newton-CG: Final Primal Objective Function value: ======= %s \n' % val_obj)
    PrimalObjValue.append(val_obj)
    print ('Newton-CG: The final relative duality gap: %s \n' % gap)
    RelativeDualityGap.append(gap)
    print ('Newton-CG: The rank of the Optimal Solution - tau*I: %s \n' % rank_x)
    print ('Newton-CG: computing time for computing preconditioners: %s \n' % prec_time)
    exectime['preconditioner'] = prec_time
    print ('Newton-CG: computing time for linear system solving (cgs time): %s \n' % pcg_time)
    exectime['pcg or minres time'] = pcg_time
    print ('Newton-CG: computing time for eigenvalue decompositions: =============== %s \n' % eig_time)
    exectime['eig_time'] = eig_time
    print ('Newton-CG: computing time used for equal weight calibration ============ %s \n' % time_used)
    exectime['total time'] = time_used 
    return x_result, y , rank_x, GradNorm, DualObjValue, PrimalObjValue, RelativeDualityGap, exectime, inneriter, rank_track, rank_positive_track

# end of the main function
######################### Test on Real Data from BNP ##########################
def reconstruct_matrix_from_csv(file_path, matrix_size, full= False):
    """
    Reconstructs a full symmetric correlation matrix from a CSV file containing the lower-left part.

    Parameters:
        file_path (str): Path to the CSV file containing the lower-left part of the matrix.
        matrix_size (int): The size of the full square matrix (number of rows/columns).

    Returns:
        np.ndarray: The reconstructed symmetric correlation matrix.
    """
    if full == True : 
        with open(file_path, 'r') as f:
            data = f.read().strip()
        values = np.array(list(map(float, data.split(',')))) 
    else: 
        # Calculate the number of values to read
        num_values = matrix_size * (matrix_size - 1) // 2
    
        # Load the required number of values from the CSV file into a 1D NumPy array
        values = []
        with open(file_path, 'r') as f:
            for line in f:
                for value in line.strip().split(','):
                    values.append(float(value))
                    if len(values) == num_values:
                        break
                if len(values) == num_values:
                    break
        values = np.array(values)
    
    # Initialize an empty matrix of the given size
    matrix = np.zeros((matrix_size, matrix_size))

    # Fill in the lower triangular part of the matrix from the file values
    index = 0
    for j in range(0, matrix_size-1):
        for i in range(j+1, matrix_size):
            matrix[i, j] = values[index]
            index += 1
    # Copy the lower triangular part to the upper triangular part to make it symmetric
    matrix += matrix.T

    # Fill the diagonal with ones
    np.fill_diagonal(matrix, 1.0)

    return matrix
# csv file contains 178 501 065 values, so the matrix has 178 501 065 * 2  off-diagonal values 
# the diagonal is size n 
# so we have n + 178 501 065 *2 = n*n => n = 18895
sizes = [18895]
GradN = dict()
DualObj = dict()
PrimalObj = dict()
RelDualityGap = dict()
IterInn = dict()
exect = dict()
track_ranks =  dict()
track_nbpositive_eig = dict()
for n in sizes : 
    
    tic = time.time()
    mat_input = reconstruct_matrix_from_csv('unrobustified_sign_correls.csv', n, full = (n == 18895))
    time_to_reconstruct_data = time.time() - tic 
    print(f"Time to reconstruct data of size n = {n}: {time_to_reconstruct_data:.4f} seconds")
    
    x_result, y, rank_x, GradNorm, DualObjValue, PrimalObjValue, RelativeDualityGap, exectime, inneriter, trackrank, trackposeig = my_correlationmatrix(mat_input, np.ones((n,1)), tau = 0, tol = 1e-6)

    #print(f"Primal variable for data of size n = {n}: {x_result} ")
    #print(f"Dual variable for data of size n = {n}: {y} ")
    GradN[n] = GradNorm
    DualObj[n] = DualObjValue
    PrimalObj[n] = PrimalObjValue
    RelDualityGap[n] = RelativeDualityGap
    exect[n] = exectime
    IterInn[n] = inneriter
    track_ranks[n] = trackrank
    track_nbpositive_eig[n] = trackposeig
    
with open("resultsAPPROX.txt", "w") as file:
    file.write("Sizes:\n")
    file.write(str(sizes) + "\n\n")
    
    file.write(f"rank of X: {rank_x}\n\n")
    
    file.write("GradN:\n")
    for key, value in GradN.items():
        file.write(f"n = {key}: {value}\n")
    file.write("\n")
    
    file.write("DualObj:\n")
    for key, value in DualObj.items():
        file.write(f"n = {key}: {value}\n")
    file.write("\n")
    
    file.write("PrimalObj:\n")
    for key, value in PrimalObj.items():
        file.write(f"n = {key}: {value}\n")
    file.write("\n")
    
    file.write("RelDualityGap:\n")
    for key, value in RelDualityGap.items():
        file.write(f"n = {key}: {value}\n")
    file.write("\n")
    
    file.write("IterInn:\n")
    for key, value in IterInn.items():
        file.write(f"n = {key}: {value}\n")
    file.write("\n")
    
    file.write("Execution Time (exect):\n")
    for key, value in exect.items():
        file.write(f"n = {key}: {value}\n")
        
    file.write("Rank of x_result:\n")  
    for key, value in track_ranks.items(): 
        file.write(f"n = {key}: {value}\n")
        
    file.write("Nb of positive eigenvalues:\n")  
    for key, value in track_nbpositive_eig.items(): 
        file.write(f"n = {key}: {value}\n")
        

