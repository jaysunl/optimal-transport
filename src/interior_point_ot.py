import numpy as np

def interior_point_ot(C, mu, nu, tol=1e-6, max_iter=100, alpha=0.95):
    """
    Solves the Optimal Transport problem using an Interior-Point Primal-Dual algorithm.
    
    Args:
        C (ndarray): Cost matrix (n x m)
        mu (ndarray): Source distribution (n,)
        nu (ndarray): Target distribution (m,)
        tol (float): Convergence tolerance
        max_iter (int): Maximum iterations
        alpha (float): Step size parameter (0 < alpha < 1)
    
    Returns:
        T (ndarray): Optimal transport plan (n x m)
    """
    n, m = C.shape

    T = np.ones((n, m)) / (n * m)

    u = np.zeros(n)  
    v = np.zeros(m) 

    t = 1.0
    mu = mu.reshape(-1, 1) 
    nu = nu.reshape(1, -1) 

    for iteration in range(max_iter):
        residual_p = np.sum(T, axis=1) - mu.flatten()  
        residual_d = np.sum(T, axis=0) - nu.flatten() 
        slackness = T * (C - u[:, None] - v[None, :])  
        
        delta_u = np.linalg.solve(np.diag(np.sum(T, axis=1)), -residual_p)
        delta_v = np.linalg.solve(np.diag(np.sum(T, axis=0)), -residual_d)

        step_size = alpha * min(1, np.min(-T[T < 0] / slackness[T < 0]))

        T += step_size * slackness
        u += step_size * delta_u
        v += step_size * delta_v

        max_residual = max(np.linalg.norm(residual_p), np.linalg.norm(residual_d))
        if max_residual < tol:
            break

        t *= 0.9  

    return T
