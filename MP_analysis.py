
# Wishart distribution PDF (for singular values squared)
from scipy.stats import wishart
import networkx as nx
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


# Path to your checkpoint file
CKPT_PATH2 = '41.ckpt'  # Change this to your .ckpt file path

CKPT_PATH = '0.ckpt'  # Change this to your .ckpt file path

# Analyze all MPNN-related weights in the checkpoint using RMT
def analyze_mpnn_weights_rmt(state_dict):
    print("\n--- MPNN Weight Spectral Analysis (RMT) ---")
    for name, param in state_dict.items():
        if 'local_model' in name and 'weight' in name:
            print(f"Analyzing {name} (MPNN weight)")
            if isinstance(param, torch.Tensor):
                matrix = param.detach().cpu().numpy()
            else:
                matrix = param
            if matrix.ndim == 2:
                # Plot singular value spectrum with Marcenko–Pastur overlay
                svals = np.linalg.svd(matrix, compute_uv=False)
                plt.figure(figsize=(6,4))
                plt.hist(svals, bins=50, alpha=0.7, density=True, label='Empirical')
                N, M = matrix.shape
                q = N / M if N <= M else M / N
                sigma = 1.0
                x = np.linspace(0, np.max(svals), 200)
                mp_pdf = marcenko_pastur_pdf(x, q, sigma)
                plt.plot(x, mp_pdf, 'k--', label='Marcenko–Pastur Law')
                plt.title(f'{name} Singular Value Spectrum')
                plt.xlabel('Singular Value')
                plt.ylabel('Density')
                plt.legend()
                plt.show()
                # If square, also plot eigenvalue spectrum with Wigner overlay
                if N == M:
                    eigvals = np.linalg.eigvals(matrix)
                    R = 2 * np.std(np.real(eigvals))
                    x_eig = np.linspace(-R, R, 200)
                    wigner_pdf = wigner_semicircle_pdf(x_eig, R)
                    plt.figure(figsize=(6,4))
                    plt.hist(np.real(eigvals), bins=50, alpha=0.7, density=True, label='Empirical')
                    plt.plot(x_eig, wigner_pdf, 'k--', label='Wigner Semicircle Law')
                    plt.title(f'{name} Eigenvalue Spectrum')
                    plt.xlabel('Eigenvalue')
                    plt.ylabel('Density')
                    plt.legend()
                    plt.show()
# Compare in_proj_weight singular value distribution to Wishart distribution
def compare_in_proj_weight_wishart(state_dict, layer=0):
    key = f'model.layers.{layer}.self_attn.in_proj_weight'
    if key not in state_dict:
        print(f"Key {key} not found in state_dict.")
        return
    W = state_dict[key]
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()
    # Only compare if 2D
    if W.ndim == 2:
        svals = np.linalg.svd(W, compute_uv=False)
        N, M = W.shape
        x = np.linspace(0, np.max(svals), 200)
        # Wishart PDF for squared singular values (approximate)
        try:
            wishart_pdf = wishart_singular_value_pdf(x, df=M, scale=np.eye(N))
            wishart_flag = True
        except Exception as e:
            wishart_pdf = np.zeros_like(x)
            wishart_flag = False

        print('wishart flag was '+str(wishart_flag))
        plt.figure(figsize=(6,4))
        plt.hist(svals, bins=50, alpha=0.7, density=True, label='Empirical')
        if wishart_flag:
            plt.plot(x, wishart_pdf, 'r-.', label='Wishart Distribution (approx)')
        plt.title(f'in_proj_weight Singular Value Spectrum vs Wishart (Layer {layer})')
        plt.xlabel('Singular Value')
        plt.ylabel('Density')
        plt.legend()
        plt.show()
    else:
        print(f'in_proj_weight is not 2D (shape: {W.shape}), skipping Wishart comparison.')
# Compare in_proj_weight eigenvalue distribution to Wigner semicircle law
def compare_in_proj_weight_wigner(state_dict, layer=0):
    key = f'model.layers.{layer}.self_attn.in_proj_weight'
    if key not in state_dict:
        print(f"Key {key} not found in state_dict.")
        return
    W = state_dict[key]
    if isinstance(W, torch.Tensor):
        W = W.detach().cpu().numpy()
    # Only compare if square
    if W.ndim == 2 and W.shape[0] == W.shape[1]:
        eigvals = np.linalg.eigvals(W)
        R = 2 * np.std(np.real(eigvals))
        x = np.linspace(-R, R, 200)
        wigner_pdf = wigner_semicircle_pdf(x, R)
        plt.figure(figsize=(6,4))
        plt.hist(np.real(eigvals), bins=50, alpha=0.7, density=True, label='Empirical')
        plt.plot(x, wigner_pdf, 'k--', label='Wigner Semicircle Law')
        plt.title(f'in_proj_weight Eigenvalue Spectrum vs Wigner Law (Layer {layer})')
        plt.xlabel('Eigenvalue')
        plt.ylabel('Density')
        plt.legend()
        plt.show()
    else:
        print(f'in_proj_weight is not square (shape: {W.shape}), skipping Wigner comparison.')

# Analyze attention weights as adjacency matrices
def analyze_attention_weights(state_dict):
    for name, param in state_dict.items():
        if 'attn' in name.lower() and 'in_proj_weight' in name:
            print(f"\nAnalyzing attention matrix as adjacency: {name}")

            # if 'in_proj_weight' in name:
            #     print('input diemnsion was: '+str(param.shape))
            q_proj_weight, k_proj_weight, v_proj_weight = param.chunk(3, dim=0)
                # print('reshaped q is shape '+str(q_proj_weight.shape))
            if isinstance(param, torch.Tensor):
                matrix = param.detach().cpu().numpy()
            else:
                matrix = param
            # Plot heatmap
            plt.figure(figsize=(6,5))
            plt.imshow(matrix, aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title(f'Attention Matrix Heatmap at End of Training') # {name}')
            plt.show()
            # Spectrum
            if matrix.ndim == 2:
                if matrix.shape[0] == matrix.shape[1]:
                    vals = np.linalg.eigvals(matrix)
                    plt.figure()
                    plt.hist(np.real(vals), bins=50, alpha=0.7)
                    plt.title(f'Attention Matrix Eigenvalue Spectrum at End of Training')
                    plt.xlabel('Eigenvalue')
                    plt.ylabel('Frequency')
                    plt.show()
                else:
                    vals = np.linalg.svd(matrix, compute_uv=False)
                    plt.figure()
                    plt.hist(vals, bins=50, alpha=0.7, density=True, label='Empirical')
                    # Overlay Marcenko–Pastur
                    N, M = matrix.shape
                    q = N / M if N <= M else M / N
                    sigma = 1.0
                    x = np.linspace(0, np.max(vals), 200)
                    mp_pdf = marcenko_pastur_pdf(x, q, sigma)
                    plt.plot(x, mp_pdf, 'k--', label='Marcenko–Pastur Law')
                    plt.title(f'Attention Matrix Singular Value Spectrum at End of Training')
                    plt.xlabel('Singular Value')
                    plt.ylabel('Density')
                    plt.legend()
                    plt.show()
            # Graph stats (for square matrices)
            if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
                # Threshold to create adjacency (optional, here >0)
                adj = (matrix > 0).astype(int)
                G = nx.from_numpy_array(adj)
                print(f"Graph stats for {name}: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, connected_components={nx.number_connected_components(G)}")



# Analyze attention weights as adjacency matrices
def compare_attention_maps(state_dict,state_dict2):
    for name, param in state_dict.items():
        if 'attn' in name.lower() and 'in_proj_weight' in name :
            print(f"\nAnalyzing attention matrix as adjacency: {name}")

            # if 'in_proj_weight' in name:
            #     print('input diemnsion was: '+str(param.shape))
            q_proj_weight, k_proj_weight, v_proj_weight = param.chunk(3, dim=0)
            print('reshaped q is shape '+str(q_proj_weight.shape))
            if isinstance(param, torch.Tensor):
                matrix = param.detach().cpu().numpy()
                matrix2 = state_dict2[name].detach().cpu().numpy()
            else:
                matrix = param
                matrix2 = state_dict2[name]
            diff_matrix = abs(matrix2 - matrix)
            # Print sum and average of all differences
            total_diff = np.sum(diff_matrix)
            avg_diff = np.mean(diff_matrix)
            print(f"Total difference: {total_diff:.6f}, Average difference: {avg_diff:.6f}")
   
            # Plot heatmap
            plt.figure(figsize=(6,5))
            plt.imshow(diff_matrix, aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title(f'Attention Matrix Heatmap Difference') # {name}')
            plt.show()


def wishart_singular_value_pdf(x, df, scale):
    # x: array of singular values (not squared)
    # To evaluate the Wishart PDF, we need a symmetric positive definite matrix
    # We'll construct a diagonal matrix from the squared singular values
    eigvals = np.square(x)
    # Create a diagonal matrix (symmetric positive definite)
    spd_matrix = np.diag(eigvals)
    # Wishart PDF expects a matrix input; for plotting, use the determinant
    # For 1D x, this is equivalent to the product of the diagonal entries
    # For visualization, return the PDF for each eigenvalue as if it were a 1x1 matrix
    pdf = np.array([wishart.pdf([[val]], df=df, scale=scale) if val > 0 else 0 for val in eigvals])
    return pdf

def load_checkpoint(ckpt_path):
    """Load a PyTorch checkpoint."""
    return torch.load(ckpt_path, map_location='cpu')

# Generate random matrices from Gaussian ensembles
def generate_goe(N):
    A = np.random.randn(N, N)
    return (A + A.T) / np.sqrt(2*N)

def generate_gue(N):
    A = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    return (A + A.conj().T) / np.sqrt(2*N)

def generate_gse(N):
    # GSE: 2N x 2N quaternion self-dual Hermitian
    A = np.random.randn(2*N, 2*N) + 1j * np.random.randn(2*N, 2*N)
    return (A + A.conj().T) / np.sqrt(4*N)

# Compare a model matrix to Gaussian ensembles
def compare_to_gaussian_ensembles(matrix, name):
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        print(f"{name} is not square, skipping GOE/GUE/GSE comparison.")
        return
    N = matrix.shape[0]
    vals_model = np.real(np.linalg.eigvals(matrix))
    vals_goe = np.real(np.linalg.eigvals(generate_goe(N)))
    vals_gue = np.real(np.linalg.eigvals(generate_gue(N)))
    vals_gse = np.real(np.linalg.eigvals(generate_gse(N//2)))
    plt.figure(figsize=(10,6))
    plt.hist(vals_model, bins=50, alpha=0.5, density=True, label=f'{name} (model)')
    plt.hist(vals_goe, bins=50, alpha=0.5, density=True, label='GOE')
    plt.hist(vals_gue, bins=50, alpha=0.5, density=True, label='GUE')
    plt.hist(vals_gse, bins=50, alpha=0.5, density=True, label='GSE')
    plt.title(f'Eigenvalue Comparison: {name} vs Gaussian Ensembles')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
# Wigner semicircle law PDF
def wigner_semicircle_pdf(x, R):
    pdf = np.zeros_like(x)
    mask = np.abs(x) <= R
    pdf[mask] = (2 / (np.pi * R**2)) * np.sqrt(R**2 - x[mask]**2)
    return pdf

def analyze_matrix_spectrum(matrix, name, plot=True):
    """Compute and plot eigenvalue distribution for a matrix."""
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    # Handle 1D matrices (vectors)
    if matrix.ndim == 1:
        # Treat as diagonal matrix for spectral analysis
        values = matrix
        title = f'Values (1D Vector): {name}'
    # For non-square matrices, use singular values
    elif matrix.shape[0] != matrix.shape[1]:
        values = np.linalg.svd(matrix, compute_uv=False)
        title = f'Singular Value Spectrum: {name}'
    else:
        values = np.linalg.eigvals(matrix)
        title = f'Eigenvalue Spectrum: {name}'
    print(f"{name}: Mean={np.mean(values):.4f}, Std={np.std(values):.4f}, Max={np.max(values):.4f}, Min={np.min(values):.4f}")
    if plot:
        plt.figure()
        plt.hist(np.real(values), bins=50, alpha=0.7)
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()

# Marcenko–Pastur PDF
def marcenko_pastur_pdf(x, q, sigma=1.0):
    b = sigma**2 * (1 + np.sqrt(q))**2
    a = sigma**2 * (1 - np.sqrt(q))**2
    pdf = np.zeros_like(x)
    mask = (x >= a) & (x <= b)
    pdf[mask] = (1 / (2 * np.pi * sigma**2 * q * x[mask])) * np.sqrt((b - x[mask]) * (x[mask] - a))
    return pdf

# New function to compare two checkpoints
def compare_checkpoints(ckpt_path1, ckpt_path2):
    """Compare the spectral properties of corresponding weight matrices from two checkpoints."""
    ckpt1 = load_checkpoint(ckpt_path1)
    ckpt2 = load_checkpoint(ckpt_path2)
    # Try to find model weights
    state_dict1 = ckpt1['model_state']
    state_dict2 = ckpt2['model_state']

    # Find common weight keys
    keys1 = set([k for k in state_dict1.keys() if k.endswith('.weight') or 'attn' in k])
    keys2 = set([k for k in state_dict2.keys() if k.endswith('.weight')or 'attn' in k])
    common_keys = keys1 & keys2
    for name in sorted(common_keys):
        print(f"Comparing {name}")
        matrix1 = state_dict1[name]
        matrix2 = state_dict2[name]
        # Only apply MP law to 2D matrices
        if isinstance(matrix1, torch.Tensor):
            matrix1 = matrix1.detach().cpu().numpy()
        if isinstance(matrix2, torch.Tensor):
            matrix2 = matrix2.detach().cpu().numpy()
        if matrix1.ndim == 2 and matrix2.ndim == 2:
            N1, M1 = matrix1.shape
            N2, M2 = matrix2.shape
            if N1 == M1 and N2 == M2:
                # Square: use eigenvalues and Wigner semicircle
                vals1 = np.linalg.eigvals(matrix1)
                vals2 = np.linalg.eigvals(matrix2)
                R1 = 2 * np.std(np.real(vals1))
                R2 = 2 * np.std(np.real(vals2))
                R = max(R1, R2)
                x = np.linspace(-R, R, 200)
                wigner_pdf = wigner_semicircle_pdf(x, R)
                plt.figure()
                plt.hist(np.real(vals1), bins=50, alpha=0.5, density=True, label='Training Start')
                plt.hist(np.real(vals2), bins=50, alpha=0.5, density=True, label='Training End')
                plt.plot(x, wigner_pdf, 'k--', label='Wigner Semicircle Law')
                plt.title(f'Comparison Spectrum (Eigenvalues): {name}')
                plt.xlabel('Value')
                plt.ylabel('Density')
                plt.legend()
                plt.show()
                # Also compare to Gaussian ensembles
                compare_to_gaussian_ensembles(matrix1, f'{name} (Checkpoint 1)')
                compare_to_gaussian_ensembles(matrix2, f'{name} (Checkpoint 2)')
            else:
                # Non-square: use singular values and Marcenko–Pastur and Wishart
                vals1 = np.linalg.svd(matrix1, compute_uv=False)
                vals2 = np.linalg.svd(matrix2, compute_uv=False)
                N, M = matrix1.shape
                q = N / M if N <= M else M / N
                sigma = 1.0  # You may want to estimate sigma from data
                x = np.linspace(0, max(np.max(vals1), np.max(vals2)), 200)
                mp_pdf = marcenko_pastur_pdf(x, q, sigma)
                # Wishart PDF for squared singular values (approximate)
                wishart_flag=False
                try:
                    wishart_pdf = wishart_singular_value_pdf(x, df=M, scale=np.eye(N))
                    print('output wishart pdf is '+str(wishart_pdf))
                    wishart_flag=True
                except Exception as e:
                    wishart_pdf = np.zeros_like(x)
                plt.hist(vals1, bins=50, alpha=0.5, density=True, label='Training Start')
                plt.hist(vals2, bins=50, alpha=0.5, density=True, label='Training End')
                plt.plot(x, mp_pdf, 'k--', label='Marcenko-Pastur Law')
                if wishart_flag:
                    plt.plot(x, wishart_pdf, 'r-.', label='Wishart Distribution (approx)')
                plt.title(f'Comparison Spectrum (Singular Values): {name}')
                plt.xlabel('Value')
                plt.ylabel('Density')
                plt.legend()
                plt.show()
        else:
            # Fallback: just compare values (for 1D)
            vals1, _ = get_spectrum(matrix1, name)
            vals2, _ = get_spectrum(matrix2, name)
            plt.figure()
            plt.hist(np.real(vals1), bins=50, alpha=0.5, label='Training Start')
            plt.hist(np.real(vals2), bins=50, alpha=0.5, label='Training End ')
            plt.title(f'Comparison Spectrum: {name}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()

# Helper to get spectrum values and title for a matrix
def get_spectrum(matrix, name):
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    if matrix.ndim == 1:
        values = matrix
        title = f'Values (1D Vector): {name}'
    elif matrix.ndim == 2:
        if matrix.shape[0] != matrix.shape[1]:
            values = np.linalg.svd(matrix, compute_uv=False)
            title = f'Singular Value Spectrum: {name}'
        else:
            values = np.linalg.eigvals(matrix)
            title = f'Eigenvalue Spectrum: {name}'
    else:
        # fallback for unexpected shapes
        values = matrix.flatten()
        title = f'Values (flattened): {name}'
    return values, title

def per_head_weights(state_dict,layer):
    attn_weights = state_dict['model.layers.'+str(layer)+'.self_attn.in_proj_weight']
    embed_dim =attn_weights.shape[1]
    num_heads =4
    head_dim = embed_dim // num_heads

    # Split Q, K, V
    qkv = attn_weights.chunk(3, dim=0)
    q_proj = qkv[0] 

    # Reshape to [num_heads, head_dim, embed_dim]
    q_proj_heads = q_proj.view(num_heads, head_dim, embed_dim)
    analyze_per_head_attention(q_proj_heads, attn_weights, head_dim)

def analyze_per_head_attention(attn_weights, attn_outputs, targets):
    """
    attn_weights: numpy array of shape (num_heads, N, N)
    attn_outputs: numpy array of shape (num_heads, N, d_model) or (num_heads, N)
    targets: numpy array of shape (N,) or (N, d_model)
    """
    num_heads = attn_weights.shape[0]
    quality_metrics = []
    r2_scores = []

    # Prepare subplots for singular value spectra

    fig, axes = plt.subplots(1, num_heads, figsize=(4*num_heads, 4), sharey=True)
    if num_heads == 1:
        axes = [axes]
    for i in range(num_heads):
        W = attn_weights[i]  # shape (head_dim, embed_dim)
        svals = np.linalg.svd(W, compute_uv=False)
        axes[i].hist(svals, bins=50, alpha=0.7, density=True, label='Empirical')
        # Overlay Marcenko–Pastur
        N, M = W.shape
        q = N / M if N <= M else M / N
        sigma = 1.0  # You may want to estimate sigma from data
        x = np.linspace(0, np.max(svals), 200)
        mp_pdf = marcenko_pastur_pdf(x, q, sigma)
        axes[i].plot(x, mp_pdf, 'k--', label='Marcenko–Pastur Law')
        axes[i].set_title(f'Head {i} Singular Value Spectrum')
        axes[i].set_xlabel('Singular Value')
        if i == 0:
            axes[i].set_ylabel('Density')
        axes[i].legend()

        # Quality metric: spectral norm
        spectral_norm = np.linalg.norm(W, ord=2)
        quality_metrics.append(spectral_norm)

    plt.tight_layout()
    plt.show()

    # Compare quality metrics side by side
    plt.figure(figsize=(6, 4))
    plt.plot(range(num_heads), quality_metrics, marker='o', label='Spectral Norm')
    plt.xlabel('Head')
    plt.ylabel('Metric')
    plt.title('Per-Head Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    #compare_checkpoints(CKPT_PATH, CKPT_PATH2)
    state_dict = load_checkpoint(CKPT_PATH)['model_state']
    state_dict2 = load_checkpoint(CKPT_PATH2)['model_state']

    #print('state_dict keys is '+str(state_dict.keys()))

    per_head_weights(state_dict2,0)
    analyze_attention_weights(state_dict2)
    print('state_dict keys is '+str(state_dict.keys()))
    print("\n--- Attention Weights as Graphs ---")
    print('state dict keys are '+str(state_dict.keys()))
    compare_attention_maps(state_dict, state_dict2)
    compare_in_proj_weight_wigner(state_dict, layer=0)
    compare_in_proj_weight_wishart(state_dict, layer=0)
    analyze_mpnn_weights_rmt(state_dict2)
    num_heads = 4



    for key,items in state_dict.items():
        print('\n\n\n name in the dictionary is: '+str(key))
        # Only analyze weight matrices (exclude bias, batch norm stats, etc.)
        if key.endswith('.weight'):
            print(f"Analyzing {key} with shape {items.shape}") #.shape}")
            analyze_matrix_spectrum( items,key)

if __name__ == '__main__':
    main()
