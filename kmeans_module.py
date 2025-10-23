
import numpy as np
import matplotlib.pyplot as plt
import os


class KMeans:


    def __init__(self, K=3, max_iters=100, tol=1e-4, random_state=42):
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.wcss = None


    # =================================================================================================#
               ### Fit the KMeans model to the input 1D data using vectorized operations. ###
    # =================================================================================================#

    def fit(self, data):
        np.random.seed(self.random_state)
        data = np.array(data).ravel()

        # Step 1: Initialize centroids randomly
        self.centroids = np.random.choice(data, self.K, replace=False)

        for iteration in range(self.max_iters):
            # --- Vectorized distance computation ---
            distances = np.abs(data[:, np.newaxis] - self.centroids[np.newaxis, :])

            # Assign each data point to nearest centroid
            labels = np.argmin(distances, axis=1)

            # --- Vectorized centroid update ---
            new_centroids = np.array([
                data[labels == k].mean() if np.any(labels == k) else self.centroids[k] # If any points are assigned to cluster k, compute their mean as new centroid Otherwise, keep the old centroid to avoid empty cluster errors

                for k in range(self.K)
            ])#.labels == k → Boolean array, True where data point belongs to cluster k.

            # Check for convergence
            shift = np.sum(np.abs(new_centroids - self.centroids))
            self.centroids = new_centroids

            if shift < self.tol:
                print(f" Converged at iteration {iteration + 1}")
                break

        self.labels = labels
        self.wcss = self._compute_wcss(data)
        return self

    # =================================================================================================#
                         ### Compute Within-Cluster Sum of Squares (WCSS). ###
    # =================================================================================================#

    def _compute_wcss(self, data):
        wcss = 0
        for k in range(self.K):
            cluster_points = data[self.labels == k]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - self.centroids[k]) ** 2)
        return wcss

    # =================================================================================================#
                   ### Predict cluster assignments for new data ###
    # =================================================================================================#

    def predict(self, data):
        data = np.array(data).ravel()
        distances = np.abs(data[:, np.newaxis] - self.centroids[np.newaxis, :])
        return np.argmin(distances, axis=1)

    # =================================================================================================#
                   ### Return model parameters (for debugging or reporting). ###
    # =================================================================================================#

    def get_params(self):
        return {
            "K": self.K,
            "max_iters": self.max_iters,
            "tol": self.tol,
            "random_state": self.random_state,
            "centroids": self.centroids,
            "wcss": self.wcss,
        }


# ============================================================
          ### Elbow Method with Auto Elbow Detection###
# ============================================================

def elbow_method(data, k_max=10, max_iters=100, tol=1e-4, random_state=42, out_dir=".", plot=True):
    """
    Compute WCSS for K = 1 to k_max, automatically find the elbow point, and plot.
    """
    data = np.array(data).ravel()
    wcss_values = []
    k_values = list(range(1, k_max + 1))

    print(f"\n--- Running Elbow Method from K=1 to K={k_max} ---")
    for k in k_values:
        model = KMeans(K=k, max_iters=max_iters, tol=tol, random_state=random_state)
        model.fit(data)
        wcss_values.append(model.wcss)
        print(f"K = {k}, WCSS = {model.wcss:.4f}")

    # --- Find elbow automatically ---
    elbow_k = find_elbow_point(k_values, wcss_values)
    print(f"\n Optimal number of clusters (Elbow point): K = {elbow_k}")

    if plot:
        elbow_plot(elbow_k, k_values, wcss_values, out_dir)

    return elbow_k, wcss_values


# ============================================================
           ### Elbow Utilities###
# ============================================================

def find_elbow_point(k_values, wcss):
    """
    Automatically find the elbow point using the geometric distance method.
    """
    x = np.array(k_values)
    y = np.array(wcss)

    # Line from first to last point
    p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    # Distance of each point from the line
    distances = np.abs(np.cross(line_vec, np.vstack([x - x[0], y - y[0]]).T)) / line_len

    # K with maximum distance = Elbow
    elbow_k = x[np.argmax(distances)]
    return elbow_k
# ============================================================
           ### Elbow plot s###
# ============================================================

def elbow_plot(elbow_k, k_values, wcss, out_dir):
    """
    Plot and save the Elbow graph highlighting the optimal K.
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, wcss, 'bo-', markersize=8)
    plt.scatter(
        elbow_k, 
        wcss[list(k_values).index(elbow_k)], 
        s=200, facecolors='none', edgecolors='r', 
        label=f'Elbow K={elbow_k}'
    )
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.title("Elbow Method")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(out_dir, "elbow_plot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f" Elbow plot saved at: {save_path}")

