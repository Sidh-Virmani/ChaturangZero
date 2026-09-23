import pytest
from p6_rule_inference.quality_metrics import compute_quality_report, play_game_for_quality
from p6_rule_inference.strategy_dna import extract_game_dna, cosine_distance
from p6_rule_inference.likelihood_estimator import compute_sequence_log_likelihood
from p6_rule_inference.bayesian_classifier import BayesianRuleClassifier
from shared.game_core.engine import ChaturangaEngine
from shared.game_core.rule_variants import RuleConfig, RuleVariant
import numpy as np

def test_quality_metrics_basic_play():
    engine = ChaturangaEngine()
    record = play_game_for_quality(engine, max_plies=10)
    assert record.plies <= 10
    assert isinstance(record.material_by_ply, list)

def test_quality_report_computes_successfully():
    config = RuleConfig.standard()
    report = compute_quality_report(config, n_games=2, verbose=False)
    assert report.rule_variant == "standard"
    assert report.n_games == 2
    assert 0.0 <= report.quality_score <= 1.0

def test_strategy_dna_extraction():
    engine = ChaturangaEngine()
    # E2 to E3, E7 to E6
    moves = ["e2e3", "e7e6"]
    dna = extract_game_dna(moves, engine)
    assert isinstance(dna, np.ndarray)
    assert len(dna) == 20  # NUM_FEATURES

def test_cosine_distance():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([2.0, 0.0, 0.0])
    assert np.isclose(cosine_distance(a, b), 1.0)
    assert np.isclose(cosine_distance(a, c), 0.0)

def test_bayesian_classifier_setup():
    classifier = BayesianRuleClassifier()
    classifier.add_model(RuleVariant.STANDARD, None)
    assert RuleVariant.STANDARD in classifier.models
    assert RuleVariant.STANDARD in classifier.engines

def test_likelihood_estimator_illegal_move_returns_neg_inf():
    engine = ChaturangaEngine()
    # e2e5 is highly illegal for padati
    moves = ["e2e5"]
    ll = compute_sequence_log_likelihood(moves, engine, None)
    assert ll == float("-inf")
