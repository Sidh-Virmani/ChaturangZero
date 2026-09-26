"""Behavioral checks for encoding, value perspective, search, and learning."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from shared.game_core import initial_state, apply_move, legal_moves, GameState, Piece, PieceKind, Side, result, GameOutcome
from shared.game_core.types import square_id_to_square as sq
from p1_representation import encode_state, encode_action, decode_action, legal_action_mask, reflect_example
from p2_neural_network import PolicyValueNet, NeuralEvaluator, save_checkpoint, load_checkpoint
from p3_mcts import MCTS, terminal_value
from p4_training.pipeline import ReplayBuffer, train_model, self_play, evaluate


def mate_position():
    return GameState((Piece('wk', PieceKind.RAJA, Side.WHITE, sq('h1')),
                      Piece('bk', PieceKind.RAJA, Side.BLACK, sq('a8')),
                      Piece('r1', PieceKind.RATHA, Side.WHITE, sq('d1')),
                      Piece('r2', PieceKind.RATHA, Side.WHITE, sq('c7'))), Side.WHITE)


class UniformEvaluator:
    def predict(self, state, moves):
        return np.full(len(moves), 1 / len(moves)), 0.0


class RepresentationTests(unittest.TestCase):
    def test_roundtrip_actions_and_masks_for_both_sides(self):
        state = initial_state()
        for _ in range(14):
            moves = legal_moves(state)
            mask = legal_action_mask(state)
            self.assertEqual(mask.sum(), len(moves))
            self.assertEqual(len({encode_action(m) for m in moves}), len(moves))
            for move in moves:
                self.assertEqual(decode_action(encode_action(move), state), move)
                self.assertTrue(mask[encode_action(move)])
            state = apply_move(state, moves[-1])

    def test_canonical_equivalent_armies_have_identical_encoding(self):
        state = initial_state()
        reflected = GameState(tuple(Piece(p.id, p.kind, p.side.opponent, (p.square[0], 7 - p.square[1]))
                                    for p in state.pieces), Side.BLACK)
        np.testing.assert_array_equal(encode_state(state), encode_state(reflected))

    def test_clock_and_repetition_are_visible_to_network(self):
        state = initial_state()
        repeated = GameState(state.pieces, state.side_to_move, halfmove_clock=50, prior_positions=(state.position_key(),))
        encoded = encode_state(repeated)
        self.assertEqual(encoded.shape, (14, 8, 8))
        self.assertTrue(np.all(encoded[12:] == .5))

    def test_reflection_preserves_actions_and_is_involution(self):
        state = initial_state()
        policy = legal_action_mask(state).astype(np.float32)
        mirrored, mirrored_policy = reflect_example(encode_state(state), policy)
        restored, restored_policy = reflect_example(mirrored, mirrored_policy)
        np.testing.assert_array_equal(restored, encode_state(state))
        np.testing.assert_array_equal(restored_policy, policy)
        move = next(m for m in legal_moves(state) if m.from_sq == sq('b1') and m.to_sq == sq('c3'))
        one_hot = np.zeros(4096); one_hot[encode_action(move)] = 1
        _, reflected = reflect_example(encode_state(state), one_hot)
        expected = (0 * 8 + 6) * 64 + (2 * 8 + 5)
        self.assertEqual(int(np.argmax(reflected)), expected)

    def test_invalid_actions_rejected(self):
        for index in [-1, 4096, 0, '22']:
            with self.assertRaises(ValueError):
                decode_action(index, initial_state())


class SearchTests(unittest.TestCase):
    def test_immediate_mate_is_found_even_with_one_simulation(self):
        state = mate_position()
        search = MCTS(UniformEvaluator(), simulations=1).search(state)
        self.assertEqual(result(apply_move(state, search.choose())), GameOutcome.WHITE_WINS)
        self.assertEqual(sum(child.visits for child in search.root.children), 1)

    def test_search_finds_mate_and_backs_up_correct_sign(self):
        state = mate_position()
        search = MCTS(UniformEvaluator(), simulations=192).search(state)
        move = search.choose()
        self.assertEqual(result(apply_move(state, move)), GameOutcome.WHITE_WINS)
        child = next(n for n in search.root.children if n.move == move)
        self.assertEqual(child.value, -1)
        self.assertGreater(search.root.value, 0)

    def test_visit_accounting_normalization_and_immutability(self):
        state = initial_state()
        search = MCTS(UniformEvaluator(), 32).search(state)
        self.assertEqual(sum(n.visits for n in search.root.children), 32)
        self.assertEqual(search.root.visits, 32)
        self.assertAlmostEqual(float(search.policy().sum()), 1, places=6)
        self.assertFalse(np.any(search.policy()[~legal_action_mask(state)]))
        self.assertEqual(len(state.move_history), 0)
        self.assertIn(search.choose(1, np.random.default_rng(7)), legal_moves(state))

    def test_terminal_search_has_no_action(self):
        state = initial_state()
        draw = GameState(state.pieces, state.side_to_move, halfmove_clock=100)
        search = MCTS(UniformEvaluator(), 20).search(draw)
        self.assertEqual(search.simulations, 0)
        self.assertEqual(search.policy().sum(), 0)
        with self.assertRaises(ValueError):
            search.choose()

    def test_seeded_root_noise_is_reproducible(self):
        a = MCTS(UniformEvaluator(), 20, seed=7).search(initial_state(), add_noise=True)
        b = MCTS(UniformEvaluator(), 20, seed=7).search(initial_state(), add_noise=True)
        np.testing.assert_array_equal(a.policy(), b.policy())


class NeuralTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = PolicyValueNet(channels=8, blocks=1)

    def test_shapes_value_range_and_real_activations(self):
        evaluator = NeuralEvaluator(self.model)
        state = initial_state()
        probabilities, value = evaluator.predict(state, legal_moves(state))
        self.assertEqual(len(probabilities), 16)
        self.assertAlmostEqual(float(sum(probabilities)), 1, places=6)
        self.assertLessEqual(abs(value), 1)
        inspected = evaluator.inspect(state)
        self.assertEqual(len(inspected['layers']), 2)
        self.assertEqual(len(inspected['layers'][0]['heatmap']), 64)
        self.assertTrue(np.isfinite(inspected['layers'][0]['heatmap']).all())

    def test_inspector_channels_and_priors_match_inference_for_both_sides(self):
        evaluator = NeuralEvaluator(self.model)
        white = initial_state()
        black = apply_move(white, legal_moves(white)[0])
        for state in (white, black):
            moves = legal_moves(state)
            probabilities, value = evaluator.predict(state, moves)
            inspected = evaluator.inspect(state)
            expected = np.zeros(64)
            for move, probability in zip(moves, probabilities):
                file, rank = move.to_sq
                expected[rank * 8 + file] += probability
            np.testing.assert_allclose(inspected['policyPriorHeatmap'], expected, atol=1e-7)
            self.assertAlmostEqual(inspected['value'], value, places=6)
            with torch.inference_mode():
                _, _, activations = self.model(torch.tensor(encode_state(state)).unsqueeze(0), True)
            for layer, activation in zip(inspected['layers'], activations):
                raw = activation[0].numpy()
                if state.side_to_move is Side.BLACK:
                    raw = raw[:, ::-1, :]
                np.testing.assert_allclose(layer['channels'], raw.reshape(8, 64), atol=1e-7)
                np.testing.assert_allclose(np.mean(np.abs(layer['channels']), axis=0),
                                           np.array(layer['heatmap']) * layer['maxActivation'], atol=1e-7)

    def test_terminal_inspection_has_no_policy_mass(self):
        state = initial_state()
        drawn = GameState(state.pieces, Side.WHITE, halfmove_clock=100)
        self.assertEqual(sum(NeuralEvaluator(self.model).inspect(drawn)['policyPriorHeatmap']), 0)

    def test_training_changes_parameters_and_reduces_fixed_batch_loss(self):
        state = initial_state()
        policy = np.zeros(4096, np.float32)
        policy[encode_action(legal_moves(state)[0])] = 1
        replay = ReplayBuffer()
        replay.extend([(encode_state(state), policy, 1.0, legal_action_mask(state))] * 8)
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        trained, losses = train_model(self.model, replay, epochs=12, batch_size=8)
        self.assertLess(losses[-1]['loss'], losses[0]['loss'])
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in trained.state_dict().items()))

    def test_checkpoint_roundtrip_preserves_predictions(self):
        encoded = torch.tensor(encode_state(initial_state())).unsqueeze(0)
        self.model.eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            save_checkpoint(path, self.model, {'trained': True})
            restored, meta = load_checkpoint(path)
            self.assertTrue(meta['trained'])
            torch.testing.assert_close(restored(encoded)[0], self.model(encoded)[0])

    def test_replay_is_bounded_and_roundtrips_without_pickle(self):
        example = (encode_state(initial_state()), np.ones(4096, np.float32) / 4096, 0, legal_action_mask(initial_state()))
        replay = ReplayBuffer(3); replay.extend([example] * 7)
        self.assertEqual(len(replay.examples), 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'replay.npz'; replay.save(path)
            restored = ReplayBuffer(); restored.load(path)
            self.assertEqual(len(restored.examples), 3)
            np.testing.assert_array_equal(restored.examples[0][0], example[0])

    def test_self_play_cap_is_explicit_and_targets_are_legal(self):
        examples, game = self_play(self.model, simulations=4, max_plies=3, seed=3)
        self.assertTrue(game['truncated'])
        self.assertEqual(game['plies'], 3)
        self.assertEqual(len(examples), 6)
        for _, policy, value, mask in examples:
            self.assertEqual(value, 0)
            self.assertAlmostEqual(float(policy.sum()), 1)
            self.assertFalse(np.any(policy[~mask]))

    def test_self_play_terminal_rewards_follow_each_players_perspective(self):
        mate = next(m for m in legal_moves(mate_position()) if m.to_sq == sq('d8'))
        with patch('p4_training.pipeline.initial_state', return_value=mate_position()), \
                patch('p3_mcts.search.SearchResult.choose', return_value=mate):
            examples, game = self_play(self.model, simulations=128, max_plies=3, seed=17)
        self.assertEqual(game['winner'], 'white')
        self.assertFalse(game['truncated'])
        self.assertEqual(examples[0][2], 1)
        self.assertEqual(examples[1][2], 1)  # Reflection keeps the outcome.

    def test_evaluation_reports_caps_and_balanced_colors(self):
        score = evaluate(self.model, games=2, simulations=2, max_plies=2)
        self.assertEqual(score['draws'], 2)
        self.assertEqual(score['truncated'], 2)
        self.assertEqual(score['score'], .5)


if __name__ == '__main__':
    unittest.main()
