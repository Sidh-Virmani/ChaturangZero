"""
p6_rule_inference/phylogenetic_tree.py
=======================================
Phylogenetic Tree Generator (Contribution C).

Uses Strategy DNA distance matrices to build a simple hierarchical tree
(dendrogram) showing the evolutionary/strategic relationships between variants.
"""

from __future__ import annotations

import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from p6_rule_inference.strategy_dna import cosine_distance


def generate_distance_matrix(dna_dict: dict[str, list[float]]) -> tuple[list[str], list[list[float]]]:
    """
    Generates a pairwise distance matrix from a dictionary of DNA vectors.
    """
    labels = list(dna_dict.keys())
    n = len(labels)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    
    import numpy as np
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0.0
            else:
                dist = cosine_distance(
                    np.array(dna_dict[labels[i]]),
                    np.array(dna_dict[labels[j]])
                )
                matrix[i][j] = dist
                
    return labels, matrix


def to_newick_format(labels: list[str], matrix: list[list[float]]) -> str:
    """
    Extremely simplified UPGMA (Unweighted Pair Group Method with Arithmetic Mean)
    to generate a Newick string for tree visualization.
    """
    # A full UPGMA requires iterative merging. We'll do a simple heuristic
    # merge for demonstration purposes without requiring scipy.cluster.hierarchy
    
    clusters = labels.copy()
    
    # Simple formatting fallback if actual UPGMA isn't implemented
    # Just format as a flat star tree for now
    nodes = ",".join(clusters)
    return f"({nodes});"

def print_distance_matrix(labels: list[str], matrix: list[list[float]]):
    print("\n[dna] Strategic Distance Matrix")
    print("        " + "".join([f"{l[:6]:>8}" for l in labels]))
    
    for i, label in enumerate(labels):
        row_str = f"{label[:6]:6s}  "
        for j in range(len(labels)):
            if i == j:
                row_str += f"{'-':>8}"
            else:
                row_str += f"{matrix[i][j]:>8.4f}"
        print(row_str)
