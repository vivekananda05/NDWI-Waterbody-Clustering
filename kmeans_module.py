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

    def fit(self, data):
        """
        Fit the KMeans model to 1D or multi-dimensional data.
        data shape: (N,) or (N, D)
        """
        np.random.seed(self.random_state)
        data = np.asarray(data, dtype="float32")

        # Ensure shape (N_samples, N_features)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples, n_features = data.shape

        # Step 1: Initialize centroids from the data points
        indices = np.random.choice(n_samples, self.K, replace=False)
        self.centroids = data[indices]  # (K, D)

        for iteration in range(self.max_iters):
            # --- Euclidean distance in D dims ---
            # diff: (N, K, D)
            diff = data[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
            distances = np.linalg.norm(diff, axis=2)  # (N, K)

            # Assign each data point to nearest centroid
            labels = np.argmin(distances, axis=1)

            # --- Update centroids ---
            new_centroids = np.empty_like(self.centroids)
            for k in range(self.K):
                cluster_points = data[labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = cluster_points.mean(axis=0)
                else:
                    # keep old centroid if cluster empty
                    new_centroids[k] = self.centroids[k]

            # Convergence check
            shift = np.sum(np.abs(new_centroids - self.centroids))
            self.centroids = new_centroids

            if shift < self.tol:
                print(f" Converged at iteration {iteration + 1}")
                break

        self.labels = labels
        self.wcss = self._compute_wcss(data)
        return self

    def _compute_wcss(self, data):
        """Compute Within-Cluster Sum of Squares (WCSS) for 1D or multi-D data."""
        data = np.asarray(data, dtype="float32")
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        wcss = 0.0
        for k in range(self.K):
            cluster_points = data[self.labels == k]
            if len(cluster_points) > 0:
                diff = cluster_points - self.centroids[k]  # (n_k, D)
                wcss += np.sum(diff ** 2)
        return wcss

    def predict(self, data):
        """Predict cluster assignments for new 1D or multi-D data."""
        data = np.asarray(data, dtype="float32")
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        diff = data[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        return np.argmin(distances, axis=1)

    def get_params(self):
        return {
            "K": self.K,
            "max_iters": self.max_iters,
            "tol": self.tol,
            "random_state": self.random_state,
            "centroids": self.centroids,
            "wcss": self.wcss,
        }


def elbow_method(data, k_max=10, max_iters=100, tol=1e-4, random_state=42, out_dir=".", plot=True):
    """
    Compute WCSS for K = 1 to k_max, automatically find the elbow point, and plot.
    Works for 1D or multi-D data.
    """
    data = np.asarray(data, dtype="float32")
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    wcss_values = []
    k_values = list(range(1, k_max + 1))

    print(f"\n--- Running Elbow Method from K=1 to K={k_max} ---")
    for k in k_values:
        model = KMeans(K=k, max_iters=max_iters, tol=tol, random_state=random_state)
        model.fit(data)
        wcss_values.append(model.wcss)
        print(f"K = {k}, WCSS = {model.wcss:.4f}")

    elbow_k = find_elbow_point(k_values, wcss_values)
    print(f"\n Optimal number of clusters (Elbow point): K = {elbow_k}")

    if plot:
        elbow_plot(elbow_k, k_values, wcss_values, out_dir)

    return elbow_k, wcss_values


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
