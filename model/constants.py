"""Core constants for POM model - only values shared across multiple files."""

# Model constants (following llama_omni2 pattern)
IGNORE_INDEX = -100
SPEECH_TOKEN_INDEX = -200
DEFAULT_SPEECH_TOKEN = "<speech>"
DEFAULT_SEP_TOKEN = "<sep>"

# Speech vocab shared between speech_lm and cosyvoice2
SPEECH_VOCAB_SIZE = 6561
