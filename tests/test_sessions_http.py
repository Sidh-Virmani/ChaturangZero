import json
import threading
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from output.server.api_server import GameSession, ApiError, Handler
from shared.game_core import GameState, Piece, PieceKind, Side, apply_move, legal_moves, result, GameOutcome, clone_state
from shared.game_core.types import square_id_to_square as sq


class SessionTests(unittest.TestCase):
    def test_presentation_scenarios_are_legal_and_resettable(self):
        session = GameSession()
        for scenario, count in [('opening', 32), ('mate', 4), ('endgame', 8)]:
            state = session.reset(scenario)
            self.assertEqual(len(state['pieces']), count)
            self.assertEqual(state['scenario'], scenario)
            self.assertEqual(state['status']['code'], 'playing')
        with self.assertRaises(ApiError):
            session.reset('unknown')

    def test_threefold_repetition_and_undo_restore_position_history(self):
        session = GameSession()
        for _ in range(2):
            for origin, destination in [('b1', 'c3'), ('b8', 'c6'), ('c3', 'b1'), ('c6', 'b8')]:
                session.play(origin, destination)
        self.assertEqual(session.snapshot()['status']['result'], 'Draw by repetition')
        self.assertEqual(session.snapshot()['selectableSquares'], [])
        self.assertEqual(session.legal_moves_from('b1'), [])
        with self.assertRaises(ApiError):
            session.play('b1', 'c3')
        restored = session.undo()
        self.assertNotEqual(restored['status']['code'], 'terminal')
        session.play('c6', 'b8')
        self.assertEqual(session.snapshot()['status']['result'], 'Draw by repetition')

    def test_clone_retains_draw_history(self):
        state = GameSession()._state
        repeated = GameState(state.pieces, state.side_to_move, prior_positions=(state.position_key(),))
        self.assertEqual(clone_state(repeated).repetition_count, 2)

    def test_no_progress_terminal_refuses_otherwise_legal_move(self):
        session = GameSession()
        session._state = GameState(session._state.pieces, Side.WHITE, halfmove_clock=100)
        with self.assertRaises(ApiError):
            session.play('e2', 'e3')
        self.assertEqual(session.snapshot()['selectableSquares'], [])

    def test_stale_revision_cannot_apply_move(self):
        session = GameSession()
        session.play('e2', 'e3', 0)
        with self.assertRaises(ApiError):
            session.play('e7', 'e6', 0)
        self.assertEqual(len(session.snapshot()['moveHistory']), 1)

    def test_reset_during_search_rejects_stale_result(self):
        entered, release = threading.Event(), threading.Event()
        class SlowEngine:
            def analyze(self, state, simulations):
                entered.set(); release.wait(5)
                return legal_moves(state)[0], {}
        session = GameSession()
        errors = []
        def search():
            try:
                session.analyze(SlowEngine(), play=True)
            except ApiError as error:
                errors.append(error)
        thread = threading.Thread(target=search)
        thread.start(); self.assertTrue(entered.wait(5))
        session.reset(); release.set(); thread.join(5)
        self.assertEqual(len(errors), 1)
        self.assertEqual(session.snapshot()['moveHistory'], [])

    def test_invalid_square_types_are_client_errors(self):
        for value in [None, 3, [], {}, True]:
            with self.assertRaises(ApiError):
                GameSession().play(value, 'e3')


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(5)

    def request(self, route, body=None, session='test-session-alpha', headers=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.url + route, data=data, headers={
            'Content-Type': 'application/json', 'X-Session-ID': session, **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def test_sessions_do_not_share_game_state(self):
        self.request('/api/reset', {})
        self.request('/api/reset', {}, session='test-session-beta')
        self.assertEqual(self.request('/api/move', {'from': 'e2', 'to': 'e3'})[0], 200)
        self.assertEqual(len(self.request('/api/state')[1]['moveHistory']), 1)
        self.assertEqual(len(self.request('/api/state', session='test-session-beta')[1]['moveHistory']), 0)

    def test_json_types_and_bounds_rejected(self):
        for route, body in [('/api/move', []), ('/api/move', {'from': 4, 'to': 'e3'}),
                            ('/api/analyze', {'simulations': 'lots'}), ('/api/analyze', {'simulations': 257}),
                            ('/api/undo', {'plies': -1})]:
            with self.subTest(route=route, body=body):
                self.assertEqual(self.request(route, body)[0], 400)

    def test_cross_origin_posts_rejected(self):
        self.assertEqual(self.request('/api/reset', {}, headers={'Origin': 'https://example.com'})[0], 403)

    def test_real_ai_analysis_and_move(self):
        self.request('/api/reset', {})
        code, payload = self.request('/api/analyze', {'simulations': 8})
        self.assertEqual(code, 200)
        self.assertEqual(payload['analysis']['simulations'], 8)
        self.assertEqual(len(payload['analysis']['layers']), 3)
        self.assertEqual(len(payload['analysis']['candidates']), 16)
        self.assertEqual(payload['state']['moveHistory'], [])
        revision = payload['state']['revision']
        code, moved = self.request('/api/ai-move', {'simulations': 8, 'revision': revision})
        self.assertEqual(code, 200)
        self.assertEqual(len(moved['state']['moveHistory']), 1)
        self.assertEqual(moved['analysis']['positionPly'], 0)
        self.assertEqual(self.request('/api/move', {'from': 'e7', 'to': 'e6', 'revision': revision})[0], 409)

    def test_invalid_session_and_unknown_route(self):
        self.assertEqual(self.request('/api/state', session='bad!')[0], 400)
        self.assertEqual(self.request('/api/not-real')[0], 404)


if __name__ == '__main__':
    unittest.main()
