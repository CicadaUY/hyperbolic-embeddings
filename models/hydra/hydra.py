#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:29:05 2025

@author: mfiori
"""

from math import acosh, pi, sqrt

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar, minimize
from numpy.random import default_rng


def hydra(D, dim=2, curvature=1, alpha=1.1, equi_adj=0.5, control=None):
    """
    Computes a hyperbolic embedding of a distance matrix using the HYDRA algorithm.

    Parameters:
        D (ndarray): Symmetric pairwise distance matrix.
        dim (int): Target embedding dimension (default 2).
        curvature (float or None): Fixed hyperbolic curvature; if None, it is optimized.
        alpha (float): Radial rescaling factor.
        equi_adj (float): Strength of equiangular angular adjustment (2D only).
        control (dict): Optional control flags for output and behavior.

    Returns:
        dict: Embedding results including radial and angular coordinates,
              and optionally stress, distances, curvature, etc.
    """
    if control is None:
        control = {}

    if curvature is not None:
        # Use fixed curvature embedding
        return hydra_fixed_curvature(D, dim, curvature, alpha, equi_adj, control)
    else:
        # Optimize curvature by minimizing stress
        control_inner = control.copy()
        control_inner["return_stress"] = True
        control_inner["return_lorentz"] = False

        eps = np.finfo(float).eps
        k_bounds = (eps, (8 / np.max(D)) ** 2)

        def stress_fn(k):
            return hydra_fixed_curvature(D, dim, k, alpha, equi_adj, control_inner)["stress"]

        result = minimize_scalar(stress_fn, bounds=k_bounds, method="bounded")
        k_opt = result.x

        # Compare to curvature = 1.0
        stress_k1 = stress_fn(1.0)
        if stress_k1 < result.fun:
            k_opt = 1.0

        out = hydra_fixed_curvature(D, dim, k_opt, alpha, equi_adj, control)
        out["curvature"] = k_opt
        return out


def hydra_fixed_curvature(D, dim=2, curvature=1, alpha=1, equi_adj=0, control=None):
    """
    Embeds the distance matrix into hyperbolic space using fixed curvature.

    Parameters:
        D (ndarray): Distance matrix.
        dim (int): Target embedding dimension.
        curvature (float): Hyperbolic curvature (> 0).
        alpha (float): Radial adjustment parameter.
        equi_adj (float): Equiangular adjustment factor in 2D.
        control (dict): Dict with optional flags for returning intermediate results.

    Returns:
        dict: Dictionary containing embedding coordinates and optionally distances/stress.
    """
    if control is None:
        control = {}

    D = np.array(D)
    np.fill_diagonal(D, 0.0)
    n = D.shape[0]

    # Default control options
    control.setdefault("return_lorentz", False)
    control.setdefault("return_dist", False)
    control.setdefault("return_stress", True)
    control.setdefault("use_eigs", False)

    if dim == 2:
        control.setdefault("isotropic_adj", True)
        control.setdefault("polar", True)
    else:
        control.setdefault("isotropic_adj", False)
        control["polar"] = False
        if equi_adj != 0.0:
            print("Warning: Equiangular adjustment only possible in dimension two.")

    # Construct hyperbolic Gram matrix
    A = np.cosh(np.sqrt(curvature) * D)
    A_max = np.max(A)

    if A_max > 1e8:
        print("Warning: Gram Matrix contains values > 1e8. Use smaller curvature or rescale distances.")
    if np.isinf(A_max):
        raise ValueError("Gram matrix contains infinite values.")

    # Spectral decomposition
    spec_vals, spec_vecs = eigh(A)
    lambda0 = spec_vals[-1]
    x0 = spec_vecs[:, -1] * np.sqrt(lambda0)  # Main Lorentz coordinate
    if x0[0] < 0:
        x0 = -x0

    # Extract lower spectrum for Euclidean coordinates
    X = spec_vecs[:, :dim]
    spec_tail = spec_vals[:dim]
    # A_frob = norm(A, 'fro')

    # Adjust magnitude of vectors based on negative spectrum
    if not control["isotropic_adj"]:
        X = X @ np.diag(np.sqrt(np.maximum(-spec_tail, 0)))

    norms = np.linalg.norm(X, axis=1)
    directional = (X.T / norms).T

    # Compute radial coordinates using Lorentzian component
    x_min = np.min(x0)
    r = np.sqrt((alpha * x0 - x_min) / (alpha * x0 + x_min))

    out = {"r": r}

    if dim == 2:
        theta = np.arctan2(X[:, 1], X[:, 0])
        if equi_adj > 0.0:
            # Equiangular adjustment of angles
            delta = 2 * pi / n
            angles = np.linspace(-pi, pi - delta, n)
            ranks = np.argsort(np.argsort(theta))
            theta_equi = angles[ranks]
            theta = (1 - equi_adj) * theta + equi_adj * theta_equi
            directional = np.stack((np.cos(theta), np.sin(theta)), axis=1)
        out["theta"] = theta

    out["directional"] = directional

    if control["return_lorentz"]:
        out["x0"] = x0
        out["X"] = X

    if control["return_dist"]:
        dist, stress = get_distance(r, directional, curvature, D)
        out["dist"] = dist
        out["stress"] = stress
    elif control["return_stress"]:
        out["stress"] = get_stress(r, directional, curvature, D)

    out["curvature"] = curvature
    out["dim"] = dim
    return out


def hyperbolic_distance(r1, r2, dir1, dir2, curvature):
    """
    Compute hyperbolic distance between two points in the Poincaré model.

    Parameters:
        r1, r2: Radial coordinates of two points.
        dir1, dir2: Directional unit vectors.
        curvature: Hyperbolic curvature.

    Returns:
        float: Hyperbolic distance between the two points.
    """
    iprod = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
    num = r1**2 + r2**2 - 2 * r1 * r2 * iprod
    denom = (1 - r1**2) * (1 - r2**2)
    acosh_arg = 1 + max(0.0, 2 * num / denom)
    return np.arccosh(acosh_arg) / np.sqrt(curvature)


def get_distance(r, directional, curvature, D=None):
    """
    Compute pairwise distances in the embedding and optionally the stress.

    Parameters:
        r: Radial coordinates of points.
        directional: Unit direction vectors.
        curvature: Hyperbolic curvature.
        D: Optional original distance matrix (to compute stress).

    Returns:
        dist_matrix: Pairwise hyperbolic distances.
        stress: Root-mean-square distance error (if D is provided).
    """
    n = len(r)
    dist_matrix = np.zeros((n, n))
    stress_sq = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            d = hyperbolic_distance(r[i], r[j], directional[i], directional[j], curvature)
            dist_matrix[i, j] = dist_matrix[j, i] = d
            if D is not None:
                stress_sq += (d - D[i, j]) ** 2

    stress = np.sqrt(stress_sq) if D is not None else None
    return dist_matrix, stress


def get_stress(r, directional, curvature, D):
    """
    Compute stress (RMS error) of embedding compared to original distances.

    Returns:
        float: stress value.
    """
    _, stress = get_distance(r, directional, curvature, D)
    return stress


def poincare_to_hyper(r, directional):
    """
    Convert Poincaré coordinates to hyperboloid model coordinates.

    Returns:
        ndarray: Hyperboloid coordinates.
    """
    return (2 * r / (1 - r**2))[:, np.newaxis] * directional


def hyper_to_poincare(X):
    """
    Convert hyperboloid model coordinates to Poincaré disk coordinates.

    Returns:
        dict with 'r', 'directional', and optional 'theta'.
    """
    norms = np.linalg.norm(X, axis=1)
    directional = (X.T / norms).T
    directional[norms == 0.0] = 0.0
    r = norms / (1 + np.sqrt(1 + norms**2))
    theta = np.arctan2(directional[:, 1], directional[:, 0]) if X.shape[1] == 2 else None
    return {"r": r, "directional": directional, "theta": theta}


def plot_hydra(
    hydra,
    labels=None,
    node_col="black",
    pch="o",
    graph_adj=None,
    crop_disc=True,
    shrink_disc=False,
    disc_col="lightgrey",
    rotation=0,
    mark_center=0,
    mark_angles=0,
    mildify=3,
    cex=1.0,
):
    """
    Plot a hyperbolic embedding in the Poincaré disk.

    Parameters:
        hydra: Output of the hydra() function.
        labels: Node labels (optional).
        node_col: Color(s) for nodes.
        pch: Marker style.
        graph_adj: Optional adjacency matrix to draw geodesics.
        crop_disc: If True, crop to max radius.
        shrink_disc: If True, shrink disc to node extent.
        disc_col: Color of background disc.
        rotation: Angle to rotate embedding (degrees).
        mark_center: If nonzero, draw cross at center.
        mark_angles: Marks on angular lines.
        mildify: Factor to adjust geodesic curves.
        cex: Size scaling factor.
    """
    fig, ax = plt.subplots()
    rotation_rad = rotation / 180 * pi
    c_radius = max(hydra["r"]) if shrink_disc else 1.0
    c_radius += 0.03 * cex

    x_nodes = hydra["r"] * np.cos(hydra["theta"] + rotation_rad)
    y_nodes = hydra["r"] * np.sin(hydra["theta"] + rotation_rad)

    ax.set_xlim(-c_radius, c_radius)
    ax.set_ylim(-c_radius, c_radius)
    ax.set_aspect("equal")
    ax.axis("off")

    disc = plt.Circle((0, 0), c_radius, color=disc_col, zorder=0)
    ax.add_artist(disc)

    if mark_center != 0.0:
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)

    # Plot geodesics
    if graph_adj is not None:
        n = len(hydra["r"])
        for i in range(n):
            for j in range(i + 1, n):
                if graph_adj[i, j] != 0:
                    r1 = hydra["r"][i] / mildify
                    r2 = hydra["r"][j] / mildify
                    dir1 = hydra["directional"][i]
                    dir2 = hydra["directional"][j]
                    X = poincare_to_hyper(np.array([r1, r2]), np.array([dir1, dir2]))
                    x0 = 1 + np.sum(X**2, axis=1)
                    lprod = x0[0] * x0[1] - np.dot(X[0], X[1])
                    dist = acosh(max(1, lprod))
                    aux = (X[1] - X[0] * lprod) / sqrt(lprod**2 - 1)
                    t = np.linspace(0, 1, 100)
                    G = outer_hyper_line(X[0], aux, dist, t)
                    P = hyper_to_poincare(G)
                    gx = P["r"] * np.cos(P["theta"] + rotation_rad) * mildify
                    gy = P["r"] * np.sin(P["theta"] + rotation_rad) * mildify
                    ax.plot(gx, gy, linestyle="dotted", color="black", linewidth=0.5)

    # Clear geodesic ends under node markers
    erase_radius = 0.03 * cex
    for x, y in zip(x_nodes, y_nodes):
        ax.add_artist(plt.Circle((x, y), erase_radius, color=disc_col, zorder=1))

    # Plot nodes
    if labels is not None:
        if isinstance(node_col, str):
            node_col = [node_col] * len(labels)
        for x, y, label, color in zip(x_nodes, y_nodes, labels, node_col):
            ax.text(x, y, str(label), color=color, fontsize=cex * 10, ha="center", va="center")
    else:
        ax.scatter(x_nodes, y_nodes, color=node_col, s=(cex * 20) ** 2, marker=pch, zorder=2)

    # Optional angular marks
    if mark_angles != 0:
        for theta in hydra["theta"]:
            x0 = (1 - mark_angles) * c_radius * np.cos(theta + rotation_rad)
            y0 = (1 - mark_angles) * c_radius * np.sin(theta + rotation_rad)
            x1 = (1 + mark_angles) * c_radius * np.cos(theta + rotation_rad)
            y1 = (1 + mark_angles) * c_radius * np.sin(theta + rotation_rad)
            ax.plot([x0, x1], [y0, y1], color="black", lw=0.5)

    plt.show()
    plt.savefig("./karateclub_hydra_embeddings.pdf", bbox_inches="tight")


def outer_hyper_line(x0, aux, dist, t):
    """
    Computes a geodesic line in the hyperboloid model.

    Parameters:
        x0 (ndarray): Starting point.
        aux (ndarray): Directional auxiliary vector.
        dist (float): Distance to cover.
        t (ndarray): Values in [0,1] parameterizing the curve.

    Returns:
        ndarray: Points along the hyperbolic geodesic.
    """
    return np.outer(np.cosh(t * dist), x0) + np.outer(np.sinh(t * dist), aux)

def hydra_plus(D, dim=2, curvature=1.0, alpha=1.1, equi_adj=0.5, control=None,
               curvature_bias=1.0, curvature_freeze=True, curvature_max=None, maxit=1000, seed=None, **kwargs):
    """
    Hydra+ with optional curvature optimization and stress minimization.

    Parameters:
        D (ndarray): Distance matrix.
        dim (int): Target embedding dimension.
        curvature (float): Initial curvature.
        alpha (float): Parameter for radial rescaling.
        equi_adj (float): Equiangular adjustment (only 2D).
        control (dict): Options for hydra.
        curvature_bias (float): Multiplier to apply to curvature before refinement.
        curvature_freeze (bool): If True, curvature is not optimized.
        curvature_max (float or None): Upper bound for curvature (auto heuristic if None).
        maxit (int): Maximum number of iterations for optimizer.
        seed (int or None): Random seed for jittering.
        **kwargs: Extra parameters passed to optimizer.

    Returns:
        dict: Optimized embedding (same format as hydra).
    """
    n = D.shape[0]
    rng = default_rng(seed)

    # Step 1: Initial HYDRA embedding
    if control is None:
        control = {}
    control["polar"] = False
    h_embed = hydra(D, dim=dim, curvature=curvature, alpha=alpha, equi_adj=equi_adj, control=control)

    # Step 2: Convert to hyperboloid coordinates and add jitter
    X0 = poincare_to_hyper(h_embed["r"], h_embed["directional"])
    x0 = X0.flatten()
    x0 += rng.normal(0, 1e-4, size=x0.shape)

    # Step 3: Handle curvature optimization
    if not curvature_freeze:
        x0 = np.append(x0, h_embed["curvature"])

    # Step 4: Set optimization bounds
    lower = np.full_like(x0, -np.inf)
    upper = np.full_like(x0, np.inf)
    if not curvature_freeze:
        if curvature_max is None:
            curvature_max = (24 / np.max(D)) ** 2
        lower[-1] = 1e-4
        upper[-1] = curvature_max

    # Step 5: Define objective and gradient
    def objective(x):
        return stress_objective(x, n, dim, D, curvature=None if not curvature_freeze else curvature_bias * h_embed["curvature"])

    def gradient(x):
        return stress_gradient(x, n, dim, D, curvature=None if not curvature_freeze else curvature_bias * h_embed["curvature"])

    # Step 6: Run optimization
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', bounds=list(zip(lower, upper)),
                      options={'maxiter': maxit, 'disp': True}, **kwargs)

    # Step 7: Convert back to Poincaré coordinates
    if not curvature_freeze:
        curv_opt = result.x[-1]
        coords = result.x[:-1].reshape(n, dim)
    else:
        curv_opt = h_embed["curvature"]
        coords = result.x.reshape(n, dim)

    poincare = hyper_to_poincare(coords)
    poincare["curvature"] = curv_opt
    poincare["curvature_max"] = curvature_max
    poincare["convergence_code"] = result.status
    poincare["stress"] = get_stress(poincare["r"], poincare["directional"], curv_opt, D)

    return poincare

def stress_objective(x, nrows, ncols, dist, curvature=None):
    if curvature is None:
        x, curvature = x[:-1], x[-1]
    x = x.reshape(nrows, ncols)
    X = x @ x.T
    u_tilde = np.sqrt(np.diag(X) + 1).reshape(-1, 1)
    H = X - u_tilde @ u_tilde.T
    D_hat = 1 / np.sqrt(curvature) * np.arccosh(np.maximum(-H, 1))
    np.fill_diagonal(D_hat, 0)
    return 0.5 * np.sum((D_hat - dist) ** 2)


def stress_gradient(x, nrows, ncols, dist, curvature=None):
    if curvature is None:
        x, curvature = x[:-1], x[-1]
        c_grad = True
    else:
        c_grad = False
    x = x.reshape(nrows, ncols)
    X = x @ x.T
    u_tilde = np.sqrt(np.diag(X) + 1).reshape(-1, 1)
    H = X - u_tilde @ u_tilde.T
    H = np.minimum(H, -1 - np.finfo(float).eps)
    D_hat = 1 / np.sqrt(curvature) * np.arccosh(-H)
    np.fill_diagonal(D_hat, 0)

    A = (D_hat - dist) / (np.sqrt(curvature * (H**2 - 1)))
    np.fill_diagonal(A, 0)
    B = (1 / u_tilde) @ u_tilde.T
    G = 2 * ((np.sum(A * B, axis=1).reshape(-1, 1) * x) - A @ x)
    z = G.flatten()

    if c_grad:
        grad_curv = -0.5 * np.sum((D_hat - dist) * D_hat) / curvature
        z = np.append(z, grad_curv)

    return z