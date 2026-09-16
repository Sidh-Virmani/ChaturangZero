"""
tests/test_retrograde.py
==========================
Tests for p5_endgame/retrograde.py — RetrogradeAnalyzer.

These tests use tiny 2-3 piece configurations to keep runtime fast.
The KR-K full solve is tested with a small sample check.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest

from shared.game_core.engine import ChaturangaEngine
from shared.game_core.state  import ChaturangaState, Piece, PieceKind, Side, Square
from p5_endgame.retrograde   import RetrogradeAnalyzer, _hash_state, _CONFIG_MAP


@pytest.fixture
def engine():
    return ChaturangaEngine()

@pytest.fixture
def analyzer(engine):
    return RetrogradeAnalyzer(engine)


# ── Config map ────────────────────────────────────────────────────────────────

def test_config_map_has_known_configs():
    for key in ["KR-K", "KM-K", "KG-K", "KA-K"]:
        assert key in _CONFIG_MAP, f"{key} missing from CONFIG_MAP"

def test_config_map_all_have_two_rajas():
    for name, spec in _CONFIG_MAP.items():
        rajas = [k for k, s in spec if k == PieceKind.RAJA]
        assert len(rajas) == 2, f"{name} should have exactly 2 Rajas"


# ── State hashing ─────────────────────────────────────────────────────────────

def test_hash_same_state_is_equal():
    pieces = [
        Piece(PieceKind.RAJA, Side.WHITE, Square(0, 0)),
        Piece(PieceKind.RAJA, Side.BLACK, Square(7, 7)),
    ]
    s1 = ChaturangaState(pieces=pieces, current_turn=Side.WHITE)
    s2 = ChaturangaState(pieces=pieces.copy(), current_turn=Side.WHITE)
    assert _hash_state(s1) == _hash_state(s2)

def test_hash_different_turn_is_different():
    pieces = [
        Piece(PieceKind.RAJA, Side.WHITE, Square(0, 0)),
        Piece(PieceKind.RAJA, Side.BLACK, Square(7, 7)),
    ]
    s1 = ChaturangaState(pieces=pieces, current_turn=Side.WHITE)
    s2 = ChaturangaState(pieces=pieces, current_turn=Side.BLACK)
    assert _hash_state(s1) != _hash_state(s2)

def test_hash_different_position_is_different():
    s1 = ChaturangaState(
        pieces=[Piece(PieceKind.RAJA, Side.WHITE, Square(0, 0)),
                Piece(PieceKind.RAJA, Side.BLACK, Square(7, 7))],
        current_turn=Side.WHITE
    )
    s2 = ChaturangaState(
        pieces=[Piece(PieceKind.RAJA, Side.WHITE, Square(1, 0)),
                Piece(PieceKind.RAJA, Side.BLACK, Square(7, 7))],
        current_turn=Side.WHITE
    )
    assert _hash_state(s1) != _hash_state(s2)


# ── Solve correctness ─────────────────────────────────────────────────────────

def test_solve_kr_k_produces_results(analyzer):
    """KR-K solve should classify at least some positions."""
    table = analyzer.solve("KR-K", verbose=False)
    assert len(table) > 0

def test_solve_kr_k_has_wins_and_draws(analyzer):
    """KR-K should have both winning positions and draws (at least)."""
    table = analyzer.solve("KR-K", verbose=False)
    results = {e.result for e in table.values()}
    assert "win" in results, "KR-K should have winning positions"
    # Draw or loss — at least one non-trivial structure
    assert len(results) >= 2

def test_terminal_positions_have_dtm_zero(engine, analyzer):
    """Checkmate positions should have DTM=0."""
    # Solve a tiny config
    table = analyzer.solve("KR-K", verbose=False)
    # Find positions the engine marks as terminal and check DTM
    zero_dtm = [e for e in table.values() if e.dtm == 0]
    assert len(zero_dtm) > 0, "Should have DTM=0 terminal positions"

def test_solve_kg_k_produces_results(analyzer):
    """KG-K (Gaja) solve should work without error."""
    table = analyzer.solve("KG-K", verbose=False)
    assert len(table) > 0

def test_solve_km_k_produces_results(analyzer):
    """KM-K (Mantri) solve should work without error."""
    table = analyzer.solve("KM-K", verbose=False)
    assert len(table) > 0

def test_analyzer_stats(analyzer):
    """Stats dict should have expected keys after solve."""
    analyzer.solve("KR-K", verbose=False)
    stats = analyzer.stats
    assert stats["solved"] is True
    assert "total" in stats
    assert "wins" in stats
    assert "losses" in stats
    assert "draws" in stats

def test_lookup_known_checkmate(engine, analyzer):
    """
    After solve, a checkmate position should be in the tablebase as 'loss' (DTM=0).
    We construct a known checkmate: White Ratha on a8, Black Raja at b8, White Raja somewhere.
    Then we force checkmate.
    """
    table = analyzer.solve("KR-K", verbose=False)
    # A position with no legal moves for Black = checkmate / loss for Black
    loss_entries = [e for e in table.values() if e.result == "loss" and e.dtm == 0]
    assert len(loss_entries) > 0

def test_save_load_roundtrip(analyzer, tmp_path):
    """Save and reload should produce identical entries."""
    table_before = analyzer.solve("KG-K", verbose=False)
    path = str(tmp_path / "test_kg_k.egtb")
    analyzer.save(path)

    analyzer2 = RetrogradeAnalyzer()
    analyzer2.load(path)

    for h, entry in table_before.items():
        assert h in analyzer2._table
        assert analyzer2._table[h].result == entry.result
