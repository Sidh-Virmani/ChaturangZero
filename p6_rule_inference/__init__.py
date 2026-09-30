"""
p6_rule_inference/__init__.py
==============================
Computational Game Archaeology - the truly unique layer of ChaturangZero-X.

This package contains four original research contributions that no published
system has applied to any historically uncertain board game:

  A. Bayesian Rule Inference
     Given a sequence of moves, compute P(rule_variant | moves).
     Files: likelihood_estimator.py, bayesian_classifier.py

  B. Rule-Conditioned Network
     One shared ResNet trunk, conditioned on a learnable rule embedding.
     The shared trunk learns rule-invariant Chaturanga strategy.
     File: multi_variant_model.py

  C. Strategy DNA Fingerprinting
     Extract a mathematical game signature from any move sequence.
     Compute strategic distance between rule variants.
     Build a phylogenetic tree of the chess family (Chaturanga->Chess->Shogi->Xiangqi).
     Files: strategy_dna.py, phylogenetic_tree.py

  D. Neural Game Quality Optimizer
     Automatically measure and optimize game quality across rule configurations.
     Discovers which rules produce the most interesting, decisive, tactical games.
     Files: quality_metrics.py, game_quality_optimizer.py
"""
