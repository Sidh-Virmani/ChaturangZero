"""Small legal positions for teaching and presenting the engine."""
from shared.game_core import GameState, Piece, PieceKind, Side, initial_state
from shared.game_core.types import square_id_to_square as sq


def make_position(name='opening'):
    if name == 'opening':
        return initial_state()
    if name == 'mate':
        return GameState((Piece('white-raja', PieceKind.RAJA, Side.WHITE, sq('h1')),
                          Piece('black-raja', PieceKind.RAJA, Side.BLACK, sq('a8')),
                          Piece('white-ratha-1', PieceKind.RATHA, Side.WHITE, sq('d1')),
                          Piece('white-ratha-2', PieceKind.RATHA, Side.WHITE, sq('c7'))), Side.WHITE)
    if name == 'endgame':
        return GameState((Piece('white-raja', PieceKind.RAJA, Side.WHITE, sq('c3')),
                          Piece('black-raja', PieceKind.RAJA, Side.BLACK, sq('g7')),
                          Piece('white-ratha', PieceKind.RATHA, Side.WHITE, sq('a2')),
                          Piece('black-ratha', PieceKind.RATHA, Side.BLACK, sq('h6')),
                          Piece('white-ashva', PieceKind.ASHVA, Side.WHITE, sq('d4')),
                          Piece('black-gaja', PieceKind.GAJA, Side.BLACK, sq('f5')),
                          Piece('white-padati', PieceKind.PADATI, Side.WHITE, sq('e5')),
                          Piece('black-padati', PieceKind.PADATI, Side.BLACK, sq('b5'))), Side.WHITE)
    raise ValueError('Choose opening, mate, or endgame')
