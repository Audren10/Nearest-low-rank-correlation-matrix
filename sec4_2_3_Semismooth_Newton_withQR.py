##   Update of the algorithm presented by Yancheng Yuan to solve the dual of 
##   the following nearest correlation matrix problem : 
##   min 0.5 || X - M ||^2 in the Frobenius norm, such that
##   X_ii = b_i and X >= tau * I (generalized constraint, classic problem with b_i = 1 and tau = 0)
##   The method comes from " A Quadratically Convergent Newton Method for Computing 
##   the Nearest Correlation Matrix Porblem" by Houduo Qi and Defeng Sun (2006)

import numpy as np
import time
import matplotlib.pyplot as plt 
import scipy.sparse.linalg as ssl
from scipy.sparse.linalg import eigsh
# cg(A, b, x0=None, *, rtol=1e-05, atol=0.0, maxiter=None, M=None, callback=None)
# minres(A, b, x0=None, *, rtol=1e-05, shift=0.0, maxiter=None, M=None, callback=None, show=False, check=False)
# bicgstab(A, b, x0=None, *, rtol=1e-05, atol=0.0, maxiter=None, M=None, callback=None)
# bicg(A, b, x0=None, *, rtol=1e-05, atol=0.0, maxiter=None, M=None, callback=None)
# cgs(A, b, x0=None, *, rtol=1e-05, atol=0.0, maxiter=None, M=None, callback=None)
# gmres(A, b, x0=None, *, rtol=1e-05, atol=0.0, restart=None, maxiter=None, M=None, callback=None, callback_type=None)[source]
times_QR = []
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
# end of my_mymexeig()
def naive_qr_shift_time(A, max_duration, tol=1e-8, Q_init=None):
    n = A.shape[0]
    A_k = A.copy()
    Q = np.eye(n) if Q_init is None else Q_init
    start_time = time.time()
    elapsed = 0
    while elapsed < max_duration:
        mu = A_k[-1, -1]
        A_shifted = A_k - mu * np.eye(n)
        Q_k, R_k = np.linalg.qr(A_shifted)
        A_k = R_k @ Q_k + mu * np.eye(n)
        Q = Q @ Q_k

        off_diag_norm = np.linalg.norm(A_k - np.diag(np.diag(A_k))) / np.linalg.norm(A_k)
        if off_diag_norm < tol:
            break
        elapsed = time.time() - start_time

    eigenvalues = np.diag(A_k)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    Q = Q[:, idx]

    return eigenvalues, Q, elapsed

def naive_qr_shift_nb(A, max_iter, tol=1e-8, Q_init=None):
    n = A.shape[0]
    A_k = A.copy()
    Q = np.eye(n) if Q_init is None else Q_init
    nb_iter = 0
    for i in range(max_iter):
        mu = A_k[-1, -1]
        A_shifted = A_k - mu * np.eye(n)
        Q_k, R_k = np.linalg.qr(A_shifted)
        A_k = R_k @ Q_k + mu * np.eye(n)
        Q = Q @ Q_k

        off_diag_norm = np.linalg.norm(A_k - np.diag(np.diag(A_k))) / np.linalg.norm(A_k)
        nb_iter+=1
        if off_diag_norm < tol:
            break

    eigenvalues = np.diag(A_k)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    Q = Q[:, idx]
    return eigenvalues, Q, nb_iter

def qr_iter_count(k, k_max, min_iter, max_iter):
    """
    Returns the number of QR iterations to use at global iteration k,
    using exponential growth from min_iter to max_iter.
    
    Parameters:
    - k: current iteration index (0-based)
    - k_max: total number of global iterations
    - min_iter: minimum QR iterations at the beginning
    - max_iter: maximum QR iterations at the end
    
    Returns:
    - int: number of QR iterations allowed at iteration k
    """
    if k_max <= 1:
        return max_iter  # fallback
    ratio = max_iter / min_iter
    iters = min_iter * (ratio ** (k / (k_max - 1)))
    return int(np.ceil(iters))

def my_mexeig(x_input, p= None, nb = 0, option = "Nbiter"):
    """
    Computation of the eigenvalues of x_input, with eigenvalues in a decreasing order 
    """
    [n, m] = x_input.shape

    if option == "Nbiter" : 
        jmax = qr_iter_count(nb, 5, 5000, 10000)
        if p is None:
            [lamb, p_x, nbtime] = naive_qr_shift_nb(x_input, max_iter=  jmax)
        else:
            #[lamb, p_x, nbtime] = naive_qr_shift_nb(x_input, Q_init = p, max_iter= jmax) #Warm Start
             [lamb, p_x, nbtime] = naive_qr_shift_nb(x_input, max_iter= jmax) #No Warm Start
            
    elif option == "Time": 
        jmax = qr_iter_count(nb, 5, 2, 10)
        if p is None:
            [lamb, p_x, nbtime] = naive_qr_shift_time(x_input, max_iter=  jmax)
        else:
            #[lamb, p_x, nbtime] = naive_qr_shift_nb(x_input, Q_init = p, max_iter= jmax) #Warm Start
             [lamb, p_x, nbtime] = naive_qr_shift_time(x_input, max_iter= jmax) #No Warm Start
            
    lam = lamb.copy()
    global times_QR
    times_QR.append(nbtime)
    print("Eigenvalue computation constraint:", nbtime, "/", jmax)
    p_x = p_x.real
    lam = lam.real
    lam[np.abs(lam) < 1e-6] = 0
    # we want eigenvalues in a decreasing order 
    if my_issorted(lam, 1): #if in increasing order, it suffices to invert the order
    
        lam = lam[::-1]
        p_x = np.fliplr(p_x)
        
    elif my_issorted(lam, -1): #in this case, it's already done 
        return p_x, lam
    else: #otherwise, we need to classify the eigenvalues and corresponding eigenvectors
        idx = np.argsort(-lam)

        lam = lam[idx]

        p_x = p_x[:, idx]

    lam = lam.reshape((n, 1))
    p_x = p_x.reshape((n, n))
    return p_x, lam

# end of my_mymexeig()


# begin of the main function
def my_correlationmatrix(M, b_input=None, tau=None, tol=None, option = "Nbiter"):
    ### Allocation of tables to stock information about the method
    
    DualObjValue = []
    PrimalObjValue = []
    RelativeDualityGap = []
    GradNorm = []
    exectime = dict()
    inneriter = []
    y_tracking = dict()
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
    
    # init
    k=0         
    f_eval = 0       # init of the number of function evaluations
    iter_whole = 200 # maximum number of the outer algo 
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
    x_result = M + np.diagflat(y)
    # ensure symmetry
    x_result = (x_result + x_result.transpose())/2.0
    # time for the eigenvalue decomposition of x_result 
    eig_time0 = time.perf_counter()
    [p_x, lamb] = my_mexeig(x_result, nb=0, option = option)  
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
    y_tracking[0] = x0
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

        y = (x0 + d).copy()
        y_tracking[k+1]=y
        #######################################################################
        x_result = M + np.diagflat(y)
        x_result = (x_result + x_result.transpose())/2.0
        eig_time0 = time.perf_counter()

        [p_x, lamb] = my_mexeig(x_result, nb = k+1, option = option)
        #[p_x, lamb] = my_mexeig_iterative(x_result, lamb)
        rank_track.append(len(lamb[lamb != 0]))
        rank_positive_track.append(len(lamb[lamb > 0]))
        eig_time = eig_time + (time.perf_counter() - eig_time0)
        [f, f_y] = dual_obj_grad(y, lamb, p_x, b_g, n)
        #######################################################################
        k_inner = 0 
        #inner loop, armijo condition 
        while k_inner <= iter_inner and f > f_0 + sigma_1*(np.power(0.5, k_inner))*slope + 1.0e-6:
            k_inner = k_inner + 1
            y = x0 + (np.power(0.5, k_inner))*d
            ###################################################################
            x_result = M + np.diagflat(y)
            x_result = (x_result + x_result.transpose())/2.0
            eig_time0 = time.perf_counter()

            [p_x, lamb] = my_mexeig(x_result, nb = k + 1 + k_inner, option = option)

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
    return x_result, y , rank_x, y_tracking, GradNorm, DualObjValue, PrimalObjValue, RelativeDualityGap, exectime, inneriter, rank_track, rank_positive_track

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
sizes = [50]

GradN = dict()
DualObj = dict()
PrimalObj = dict()
RelDualityGap = dict()
IterInn = dict()
exect = dict()
ysol = dict()
track_ranks =  dict()
track_nbpositive_eig = dict()

for n in sizes : 
    
    tic = time.time()
    mat_input = reconstruct_matrix_from_csv('unrobustified_sign_correls.csv', n, full = (n == 18895))
    time_to_reconstruct_data = time.time() - tic 
    print(f"Time to reconstruct data of size n = {n}: {time_to_reconstruct_data:.4f} seconds")
    ## choose your option : "Nbiter" or "Time"
    x_result, y, rank_x, y_tracking, GradNorm, DualObjValue, PrimalObjValue, RelativeDualityGap, exectime, inneriter, trackrank, trackposeig = my_correlationmatrix(mat_input, np.ones((n,1)), tau = 0, tol = 1e-6, option="Nbiter")
    #print(f"Primal variable for data of size n = {n}: {x_result} ")
    #print(f"Dual variable for data of size n = {n}: {y} ")
    
    GradN[n] = GradNorm
    DualObj[n] = DualObjValue
    PrimalObj[n] = PrimalObjValue
    RelDualityGap[n] = RelativeDualityGap
    exect[n] = exectime
    IterInn[n] = inneriter
    ysol[n] = y_tracking
    track_ranks[n] = trackrank
    track_nbpositive_eig[n] = trackposeig

with open("resultsQR1000.txt", "w") as file:
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
        
import numpy as np

with open("ydualQR1000.txt", "w") as file:
    for key, inner_dict in ysol.items():
        file.write(f"n = {key}:\n")
        # Trier les éléments du dictionnaire par clé (pour garantir l'ordre)
        sorted_items = sorted(inner_dict.items(), key=lambda item: item[0])

        # Écrire les valeurs individuelles de y
        file.write("Valeurs de y :\n")
        for iter_key, y_val in sorted_items:
            file.write(f"  Itération {iter_key} : {y_val}\n")

        # Calculer les différences entre itérations successives
        if len(sorted_items) >= 2:
            file.write("Différences entre itérations (y_i - y_{i-1}) :\n")
            for i in range(1, len(sorted_items)):
                iter_prev, y_prev = sorted_items[i - 1]
                iter_curr, y_curr = sorted_items[i]
                # Conversion en tableaux numpy pour effectuer les calculs
                y_prev_arr = np.array(y_prev)
                y_curr_arr = np.array(y_curr)
                delta_y = y_curr_arr - y_prev_arr
                # Calcul des statistiques pour delta_y
                mean_delta = np.mean(delta_y)
                var_delta = np.var(delta_y)
                min_delta = np.min(delta_y)
                max_delta = np.max(delta_y)
                # Écriture des résultats dans le fichier
                file.write(f"  Différence entre itération {iter_curr} et {iter_prev} : {delta_y}\n")
                file.write(f"    Moyenne : {mean_delta}\n")
                file.write(f"    Variance : {var_delta}\n")
                file.write(f"    Minimum : {min_delta}\n")
                file.write(f"    Maximum : {max_delta}\n")
        else:
            file.write("Pas assez de valeurs pour calculer les différences.\n")

        file.write("\n")
