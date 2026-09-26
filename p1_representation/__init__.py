"""Canonical, side-to-move board encoding and a bijective 4096-action space."""
from .encoding import ACTION_SIZE, CHANNELS, KINDS, encode_state, encode_action, decode_action, legal_action_mask, reflect_example
