"""Canonical label mapping for the HandSpeak 26-class (A-Z) model.

The combined dataset produced by :mod:`preprocessing.combine_26_class` uses
**contiguous labels 0-25** where ``0 -> 'A'``, ``1 -> 'B'``, ..., ``25 -> 'Z'``.
This is the single source of truth shared by preprocessing, training, and
inference. Any earlier 24-class mapping (which skipped J and Z) is obsolete and
must not be used, because the trained model has 26 output neurons.

The functions below intentionally keep the same names/behaviour as the old
24-class module so call sites do not break, but every conversion is now the
identity over the full A-Z alphabet.
"""

from utils.config import NUM_CLASSES

# All 26 ASL letters in model-output order (A=0 ... Z=25).
LETTERS = [chr(ord("A") + i) for i in range(26)]

# In the combined dataset the raw label equals the contiguous class index.
# 0 -> A, 1 -> B, ..., 9 -> J, ..., 25 -> Z.
SIGN_LABELS = list(range(26))

# Ensure the mapping stays consistent with the configured number of classes.
assert len(SIGN_LABELS) == NUM_CLASSES, "SIGN_LABELS length must match NUM_CLASSES"
assert len(LETTERS) == NUM_CLASSES, "LETTERS length must match NUM_CLASSES"


def label_to_letter(raw_label):
    """Convert a dataset label (0-25) to its English letter.

    Example: 0 -> 'A', 9 -> 'J', 25 -> 'Z'.
    """
    return LETTERS[int(raw_label)]


def label_to_index(raw_label):
    """Convert a dataset label (0-25) to the contiguous class index.

    In the combined dataset these are identical.
    """
    return int(raw_label)


def index_to_label(class_index):
    """Convert a contiguous class index (0-25) back to a dataset label.

    In the combined dataset these are identical.
    """
    return int(class_index)


def index_to_letter(class_index):
    """Convert a contiguous class index (0-25) to its English letter."""
    return LETTERS[int(class_index)]


def get_all_letters():
    """Return the 26 class letters in model-output order (A-Z)."""
    return list(LETTERS)
