
# %% This code is used to demonstrate the 2D BCS method
# Reference: Ji et al(2007, 2008 & 2009), Zhao et al (2018), & Tipping(2021)

import numpy as np
import pandas as pd
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from scipy.linalg import pinv, inv

def TwoD_BCS(ExData, hy, hx, rx, ry):
    """
    Main routine for 2D Bayesian Compressive Sensing.
    
    Parameters:
        ExData (DataFrame): DataFrame containing UPV experiment measurement data;
                with three columns:[Y-coordinate, X-coordinate, measured value].
        hy (float): Height of the domain.
        hx (float): Width of the domain.
        rx (float): Grid spacing in x–direction.
        ry (float): Grid spacing in y–direction.
    
    Returns:
        Tuple containing:
          - Reconstructed field RC_field,
          - Estimated weight vector Muw,
          - Final alpha hyperparameters,
          - Active index set,
          - Reconstructed coefficient matrix Omg,
          - ExData2d (the 2D measurement matrix),
    """
    # Convert to numpy array and sort by second column (x-coordinate)
    ExData = ExData.values
    ExData = ExData[ExData[:, 1].argsort()]
    Mtot = ExData.shape[0]

    # Define grid coordinates (rounded to 2 decimals)
    y_coords = np.round(np.arange(0, hy + ry, ry), 2)
    x_coords = np.round(np.arange(0, hx + rx, rx), 2)
    N1, N2 = len(y_coords), len(x_coords)

    # Build lookup dictionaries for coordinate indices
    y_index_map = {val: idx for idx, val in enumerate(y_coords)}
    x_index_map = {val: idx for idx, val in enumerate(x_coords)}
    # Use list comprehension to obtain measurement indices as a 2D array
    IND = np.array([[y_index_map[y_val], x_index_map[x_val]] for y_val, x_val, _ in ExData])

    # Get unique indices (vertical and horizontal) from measurements
    unique_y = np.unique(IND[:, 0])
    unique_x = np.unique(IND[:, 1])

    # Build DCT matrices and compute their (Moore–Penrose) inverses
    W1 = dct(np.eye(N1), type=2, norm='ortho').T
    W2 = dct(np.eye(N2), type=2, norm='ortho').T
    B1 = pinv(W1)
    B2 = pinv(W2).T

    # Truncate to a number of coefficients determined by sqrt(Mtot)
    trunc = 2 * int(np.ceil(np.sqrt(Mtot)))
    A1 = B1[unique_y, :trunc]
    A2 = B2[:trunc, unique_x]

    eta = 1E-6
    ExDataVal = ExData[:, 2]  # measurement values

    # Call the “BCS Core” routine that performs the core BCS steps.
    J, Vtr, active_idx, alpha, ExData2D, Nc1, Nr2 = BCS_Core(ExDataVal, eta, Mtot, B1, B2, A1, A2, IND)

    # Compute the weight vector Muw for the active basis
    Muw = inv(J + np.diag(alpha)) @ Vtr

    # Rearrange the coefficients into the Omg matrix.
    # Note: Here we assume that the active indices relate to the 2D layout in a block fashion.
    IN = np.ceil((active_idx + 1) / Nc1).astype(int)
    Omg = np.zeros((Nc1, Nr2))
    for i in range(1, Nr2 + 1):
        idx = np.where(IN == i)[0]
        # Map global indices to block–coordinates
        local_idx = active_idx[idx] - (i - 1) * Nc1
        Omg[local_idx, i - 1] = Muw[idx].flatten()

    # Reconstruct the full field from the computed coefficients.
    RC_field = B1[:, :Nc1] @ Omg @ B2[:Nr2, :]

    plot_2D_surface(x_coords, y_coords, RC_field, ExData, hy, hx)

    return RC_field, Muw, alpha, active_idx, Omg, ExData2D

def BCS_Core(ExDataVal, eta, Mtot, B1, B2, A1, A2, IND):
    """
    BCS core routine for Bayesian Compressive Sensing.
    Constructs the measurement matrix, computes the projection PHI and iterates to update
    the hyperparameters alpha.
    
    Returns many intermediate variables needed for later reconstruction.
    """
    # Build a 2D measurement matrix from the 1D measurement vector
    N1, N2 = B1.shape[0], B2.shape[0]
    ExData2D = np.zeros((N1, N2))
    for i in range(Mtot):
        ExData2D[IND[i, 0], IND[i, 1]] = ExDataVal[i]
    # Remove rows and columns that are completely zero
    ExData2D = ExData2D[~np.all(ExData2D == 0, axis=1), :]
    ExData2D = ExData2D[:, ~np.all(ExData2D == 0, axis=0)]

    # Get indices of the nonzero (and zero) entries in ExData2D
    nonzero_rows, nonzero_cols = np.where(ExData2D != 0)
    # Sort indices based on nonzero_cols
    sorted_indices = np.argsort(nonzero_cols)
    nonzero_rows = nonzero_rows[sorted_indices]
    nonzero_cols = nonzero_cols[sorted_indices]

    zero_rows, zero_cols = np.where(ExData2D == 0)
    # Sort indices based on zero_cols
    sorted_indices = np.argsort(zero_cols)
    zero_rows = zero_rows[sorted_indices]
    zero_cols = zero_cols[sorted_indices]

    # Precompute sizes from A1 and A2
    Nr1, Nc1 = A1.shape; Nr2, Nc2 = A2.shape

    # Build the projection matrix PHI and auxiliary structure At.
    # For each pair (i, j), the corresponding column in PHI is given by
    # the outer product of A1[:, i] and A2[j, :] evaluated at the nonzero positions.
    PHI_columns = []
    # We store each “submatrix” for later use. Instead of using a dictionary for each entry,
    # we store the submatrix in a 2D list (which we later flatten).
    At = [[None] * Nc1 for _ in range(Nr2)]
    for j in range(Nr2):
        for i in range(Nc1):
            # Compute the outer product (submatrix)
            submatrix = np.outer(A1[:, i], A2[j, :])
            # Extract the entries corresponding to the nonzero measurement positions.
            col = (A1[nonzero_rows, i] * A2[j, nonzero_cols]).reshape(-1, 1)
            PHI_columns.append(col)
            # Zero out the entries corresponding to the zero locations.
            # (This might be redundant if those entries are already zero in ExData2D.)
            submatrix[zero_rows, zero_cols] = 0
            At[j][i] = submatrix
    PHI = np.hstack(PHI_columns)

    # Initialize hyperparameter alpha via a simple heuristic.
    PHI2 = np.sum(PHI**2, axis=0)
    PHIt = PHI.T @ ExDataVal
    ratio = (PHIt**2) / PHI2
    init_idx = np.argmax(np.abs(ratio))
    alpha = 1.0  # initial scalar value (will become a vector as indices are added)
    c = 1  # constant in gamma priors (set to 1)

    # Run iterative update on alpha and record the signal evolution.
    # Note: We use a fixed maximum number of iterations (here 10000) and a stopping tolerance.
    Signal = np.ones((PHI.shape[0], 10000)) * 1e4  # large initial value
    active_idx = np.array([init_idx], dtype=int)  # start with the index maximizing the ratio

    for count in range(10000):
        # Compute d using the current active PHI column.
        if np.isscalar(alpha):
            phi_active = PHI[:, active_idx].reshape(-1, 1)
            inv_term = inv(np.eye(PHI.shape[0]) + np.outer(phi_active, phi_active.T)/alpha)
        else:
            phi_active = PHI[:, active_idx]
            inv_term = inv(np.eye(PHI.shape[0]) + phi_active @ inv(np.diag(alpha)) @ phi_active.T)

        d = (c / PHI.shape[0]) * (ExDataVal.T @ inv_term @ ExDataVal)

        # Update alpha and active index via the dedicated routine.
        alpha, active_idx = alpha_iteration(PHI, ExDataVal, c, d, eta)

        # Precompute J and Vtr based on the stored At.
        n_active = len(active_idx)
        J = np.zeros((n_active, n_active))
        Vtr = np.zeros((n_active, 1))
        # Flatten At to access submatrices corresponding to each PHI column.
        At_flat = np.array([At[j][i] for j in range(Nr2) for i in range(Nc1)])
        for i in range(n_active):
            for j in range(n_active):
                # The “trace” inner product of submatrix i and j.
                J[i, j] = np.trace(At_flat[active_idx[i]] @ At_flat[active_idx[j]].T)
            Vtr[i, 0] = np.trace(ExData2D @ At_flat[active_idx[i]].T)

        # Compute current weight estimate and reconstructed signal
        Muw = inv(J + np.diag(alpha)) @ Vtr
        Signal[:, count] = (PHI[:, active_idx] @ Muw).reshape(-1)

        # Check for convergence (relative change in the reconstructed signal)
        if count > 1:
            rel_change = np.linalg.norm(Signal[:, count] - Signal[:, count - 1]) / np.linalg.norm(Signal[:, count - 1])
            if rel_change < 0.001:
                break

    return J, Vtr, active_idx, alpha, ExData2D, Nc1, Nr2

import numpy as np
from numpy.linalg import inv, pinv

def alpha_iteration(PHI, ExDataVal, c, d, eta):
    """
    Perform iterative re-estimation of the hyperparameters (alpha) and the active set of coefficients.
    This routine attempts to maximize the marginal likelihood by updating alpha for re-estimation,
    adding new coefficients, or deleting coefficients as appropriate.
    
    Parameters:
        PHI (np.ndarray): The projection (or design) matrix with shape (M, N).
        ExDataVal (np.ndarray): The measurement vector with shape (M,).
        c (float): Gamma-prior parameter (typically set to 1).
        d (float): Gamma-prior parameter.
        eta (float): Convergence threshold for the relative change in the marginal likelihood.
        max_outer_iter (int): Maximum number of outer iterations.
    
    Returns:
        alpha (np.ndarray): 1D array of hyperparameters corresponding to the active coefficients.
        active_idx (np.ndarray): 1D array of indices (columns of PHI) corresponding to non–trivial coefficients.
    """
    max_outer_iter=10000

    M, N = PHI.shape

    # Precompute quantities needed for initialization.
    # K: effective degrees-of-freedom vector.
    K = np.full(N, M + 2 * c)

    # At: inner products between each column of PHI and the measurement vector ExDataVal.
    Ati = PHI.T @ ExDataVal  # shape: (N,)

    # A2: squared norm of each column of PHI.
    A2i = np.sum(PHI**2, axis=0)  # shape: (N,)

    # G2: a scalar computed from ExDataVal then expanded to a vector.
    G2_scalar = (ExDataVal.T @ ExDataVal) + 2 * d
    G2 = np.full(N, G2_scalar)

    # Compute a “marginal likelihood” proxy for each candidate coefficient.
    X = (G2 * A2i) / (Ati**2)
    ml = K * np.log(X / (K)) - (K - 1) * np.log((X - 1) / (K - 1))

    # Select an initial active index from those with a high marginal likelihood.
    ml_sum = ml.copy()  # ml is 1D here.
    while True:
        idx_init = np.argmax(ml_sum)
        denom = A2i[idx_init] * (A2i[idx_init] - Ati[idx_init]**2 / (G2[idx_init]))
        num = K[idx_init] * Ati[idx_init]**2 / (G2[idx_init]) - A2i[idx_init]
        alpha_init = 1.0 / (num / denom)
        if alpha_init > 0:
            break
        else:
            ml_sum[idx_init] = -np.inf  # Remove this index from further consideration.

    # Initialize the active set and corresponding hyperparameters.
    active_idx = [idx_init]
    alpha_vec = np.array([alpha_init])  # Active hyperparameters (for now a 1-element vector)

    # Initialize the effective “dictionary” (active columns) and compute the posterior covariance and mean.
    Phi_active = PHI[:, active_idx]         # (M x r), where r is the number of active coefficients.
    Hessian = np.diag(alpha_vec) + (Phi_active.T @ Phi_active)  # (r x r)
    Sig = inv(Hessian)                      # Posterior covariance (r x r)
    mu = Sig @ (Phi_active.T @ ExDataVal)          # Posterior mean (r,)

    # For all columns (even inactive) we will compute auxiliary quantities.
    # left: inner product between every column of PHI and the current active dictionary.
    left = (PHI.T @ Phi_active).flatten()  # For one active coefficient, this is (N,). For r>1, adjust below.
    # For simplicity, here we assume that for r>1 we can compute a weighted version; in many applications
    # one loops over each active coefficient. In the following we update S, Q, G for each column of PHI.
    S = A2i - (np.sum(Sig * (left ** 2)) if np.isscalar(Sig) else np.sum(Sig * ((left)**2), axis=0))
    Q = Ati - (np.sum(Sig * Ati[active_idx] * left) if np.isscalar(Sig) else np.sum((Sig * Ati[active_idx] * left), axis=0))
    G = G2 - (np.sum(Sig * Ati[active_idx]**2) if np.isscalar(mu) else np.sum(Sig * (Ati[active_idx])**2))

    # Initialize an array to record the marginal likelihood at each outer iteration.
    ML_count = np.zeros(max_outer_iter)

    # Begin outer iteration: update hyperparameters (re-estimation, addition, deletion).
    for count in range(max_outer_iter):
        # Make copies of S, Q, and G to update candidate quantities.
        s = S.copy()
        q = Q.copy()
        g = G.copy()

        # For indices in the current active set, update candidate quantities.
        # (Here we loop over the active set. For large r, you may wish to vectorize further.)
        for j, idx in enumerate(active_idx):
            denom_j = alpha_vec[j] - S[idx]
            s[idx] = alpha_vec[j] * S[idx] / denom_j
            q[idx] = alpha_vec[j] * Q[idx] / denom_j
            g[idx] = G[idx] + (Q[idx]**2) / (denom_j)

        # Compute a candidate update for alpha for every coefficient.
        # Here ck1 is computed elementwise.
        ck1 = (K * q**2 / (g) - s) / (s * (s - q**2 / (g)))
        # Use 1/ck1 as the candidate new alpha (theta)
        theta = 1.0 / (ck1)

        # Prepare an array for updated marginal likelihood values.
        ml_new = np.full(N, -np.inf)

        # -- Re-estimation (for coefficients already in the active set) --
        active_set = np.array(active_idx)
        ire = np.intersect1d(np.where(theta > 0)[0], active_set)
        if ire.size > 0:
            for idx in ire:
                j = np.where(active_set == idx)[0][0]  # position in the active set
                Alpha1 = theta[idx]
                Alpha0 = alpha_vec[j]
                delta = 1.0 / Alpha1 - 1.0 / Alpha0
                # Compute a candidate marginal likelihood update.
                ml_new[idx] = ((K[idx] - 1) * np.log(1 + S[idx] * delta) +
                               K[idx] * np.log( ((Alpha0 + s[idx]) * g[idx] - q[idx]**2) *
                                                Alpha1 /
                                                (((Alpha1 + s[idx]) * g[idx] - q[idx]**2) * Alpha0)))

        # -- Addition (for coefficients with positive theta that are not yet active) --
        iad = np.setdiff1d(np.where(theta > 0)[0], active_set)
        if iad.size > 0:
            ml_new[iad] = np.log(theta[iad] / (theta[iad] + s[iad])) - \
                          K[iad] * np.log(1 - (q[iad]**2 / (g[iad])) / (theta[iad] + s[iad]))
        
        # -- Deletion (for currently active coefficients that now yield nonpositive candidate theta) --
        not_in_ig = np.setdiff1d(np.arange(N), np.where(theta > 0)[0])
        ide = np.intersect1d(not_in_ig, active_set)
        if ide.size > 0:
            for idx in ide:
                j = np.where(active_set == idx)[0][0]
                Alpha_val = alpha_vec[j]
                ml_new[idx] = -np.log(1 - S[idx] / (Alpha_val)) - \
                              K[idx] * np.log(1 + Q[idx]**2 / (G[idx] * (Alpha_val - S[idx])))

        # Record the overall (maximum) marginal likelihood value.
        ML_count[count] = np.max(ml_new)
        idx_star = np.argmax(ml_new)

        # Check convergence: relative change in ML is small.
        if count > 1 and np.abs(ML_count[count] - ML_count[count - 1]) < (np.max(ML_count) - ML_count[count]) * eta:
            break

        # --- Update step: re-estimate, add, or delete coefficient corresponding to idx_star ---
        if theta[idx_star] > 0:
            # Case 1: Re-estimation or addition.
            if idx_star in active_set:
                # Re-estimation: update alpha for the active coefficient.
                j = np.where(active_set == idx_star)[0][0]
                Alpha_new = theta[idx_star]
                delta = Alpha_new - alpha_vec[j]
                # Perform one (or more) inner update iterations.
                # (The following update uses a scalar update for the j-th coefficient.)
                Sig_jj = Sig[j, j] if Sig.ndim == 2 else Sig
                mu_j = mu[j] if mu.ndim > 0 else mu
                kappa = delta / (1 + Sig_jj * delta)
                # Update the posterior mean and covariance (rank-1 update).
                mu = mu - kappa * mu_j * (Sig[:, j] if Sig.ndim == 2 else Sig)
                if Sig.ndim == 2:
                    Sig = Sig - kappa * np.outer(Sig[:, j], Sig[:, j])
                else:
                    Sig = Sig - kappa * (Sig**2)
                comm = PHI.T @ (Phi_active @ Sig[:, j])
                S[:,] = S[:,] + kappa * (comm ** 2)
                Q[:,] = Q[:,] + kappa * mu_j * comm
                G[:,] = G[:,] + kappa * (Sig[:, j].T @ Ati[active_set,]) ** 2                
                alpha_vec[j] = Alpha_new
            else:
                # Addition: add the new coefficient idx_star to the active set.
                Alpha_new = theta[idx_star]
                # Expand the active set.
                active_set = np.append(active_set, idx_star)
                alpha_vec = np.append(alpha_vec, Alpha_new)
                # Active dictionary.
                Phi_activei = PHI[:, idx_star]
                # Recompute the posterior covariance and mean for the updated set.
                Hessiani = Alpha_new + S[idx_star]
                Sigi = 1/(Hessiani)
                mui = Sigi * Q[idx_star]
                if isinstance(Sig, float):
                    sk1 = Sig * (Phi_active.T @ Phi_activei)
                    sk2 = Phi_activei - Phi_active * sk1
                else:
                    sk1 = Sig @ (Phi_active.T @ Phi_activei)
                    sk2 = Phi_activei - Phi_active @ sk1
                sk3     = -Sigi * sk1
                if isinstance(sk3, float):
                    Sig = np.block([[Sig + Sigi * np.outer(sk1, sk1), sk3], [sk3, Sigi]])
                else:
                    Sig = np.block([[Sig + Sigi * np.outer(sk1, sk1), sk3[:, None]], [sk3[None, :], Sigi]])
                mu  = np.append([mu - mui * sk1], [mui])
                sk4 = PHI.T @ sk2
                S[:,] = S[:,] - Sigi * (sk4 ** 2)
                Q[:,] = Q[:,] - mui * sk4
                G[:,] = G[:,] - Sigi * (ExDataVal.T @ sk2) ** 2
                Phi_active = np.column_stack([Phi_active, Phi_activei])
            # Update the active set list.
            active_idx = list(active_set)
        else:
            # Case 2: Deletion -- remove idx_star from the active set.
            if idx_star in active_set:
                j = np.where(active_set == idx_star)[0][0]
                Sigii = Sig[j, j]
                mui   = mu[j]
                Sigi  = Sig[:, j]
                Sig   = Sig - np.outer(Sigi, Sigi) / Sigii
                Sig   = np.delete(Sig, j, axis=0)
                Sig   = np.delete(Sig, j, axis=1)
                mu    = mu - mui / Sigii * Sigi
                mu    = np.delete(mu, j)
                sk5   = PHI.T @ (Phi_active @ Sigi)
                S[:,] = S[:,] + (sk5 ** 2) / Sigii
                Q[:,] = Q[:,] + mui / Sigii * sk5
                G[:,] = G[:,] + (Sigi.T @ Ati[active_idx,]) ** 2 / Sigii
                Phi_active = np.delete(Phi_active, j, axis=1)
                active_idx = np.delete(active_idx, j)
                alpha_vec  = np.delete(alpha_vec, j)

    return alpha_vec, np.array(active_idx)

def plot_2D_surface(x_coords, y_coords, RC_field, ExData, hy, hx):
    """
    Plot the 2D surface realization.
    
    Parameters:
        x_coords (ndarray): x–coordinate grid (1D).
        y_coords (ndarray): y–coordinate grid (1D).
        RC_field (ndarray): Reconstructed field (2D).
        ExDataVal (ndarray): Original measurement data (with [y, x, value]).
        hy (float): Height of the domain.
        hx (float): Width of the domain.
    """
    Xi, Yi = np.meshgrid(x_coords, y_coords)
    fig, ax = plt.subplots(figsize=(10, 8))
    c = ax.pcolormesh(Xi, Yi, RC_field, shading='auto', cmap='jet')
    ax.set_aspect('equal')
    ax.set_xlim([0, hx])
    ax.set_ylim([0, hy])
    ax.set_title('Material property distribution in 2D', pad=20)
    ax.set_xlabel('X - axis (cm)')
    ax.set_ylabel('Y - axis (cm)')

    # Plot measurement points
    ax.scatter(ExData[:, 1], ExData[:, 0], c='k', s=50, label='Measurement points')
    # Annotate the measurement values
    for (y_val, x_val, val) in ExData:
        ax.annotate(f'{val:.2f}', (x_val, y_val),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=8, color='black')
    fig.colorbar(c, ax=ax, label='E (GPa)')
    ax.invert_yaxis()  # Reverse Y–axis
    ax.xaxis.tick_top()  # Move X–axis ticks to the top
    ax.xaxis.set_label_position('top')
    ax.grid(False)
    fig.tight_layout()
    plt.show()

# ===== Example usage =====
if __name__ == '__main__':
    # Load data (adjust the file path as necessary)
    data_path = r'C:\Sample_Data_UPV_Experiment_DataA1.xlsx'
    ExData = pd.read_excel(data_path, header=None)
    # Define domain and grid parameters
    hy = 41.5; hx = 19.75; ry = 0.25; rx = 0.25
    # Run the 2D BCS algorithm
    RC_field, Muw, alpha, active_idx, Omg, ExData2D = TwoD_BCS(ExData, hy, hx, rx, ry)