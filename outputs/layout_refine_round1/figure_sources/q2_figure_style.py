"""Q2 figure semantics: positive = improvement; negative = deterioration.
MAE improvement is reference MAE minus comparison MAE. Never use value-zero
or native feature zeros to change the frozen data/missing protocol.
"""
from matplotlib.colors import LinearSegmentedColormap
NEGATIVE_COLOR = "#BF8580"  # low-saturation warm red
ZERO_COLOR = "#FFFFFF"
POSITIVE_COLOR = "#357F86"  # restrained blue-green

def performance_cmap():
    # Odd lookup-table length makes the scientific zero exactly white.
    return LinearSegmentedColormap.from_list(
        "q2_negative_red_zero_white_positive_teal",
        [NEGATIVE_COLOR, ZERO_COLOR, POSITIVE_COLOR], N=257)

def sequential_cmap():
    return LinearSegmentedColormap.from_list(
        "q2_confusion_sequential_blue", ["#F7FBFF", "#C6DDF0", "#5B9BD5", "#084B83"], N=256)

def delta_color(value):
    return POSITIVE_COLOR if value > 0 else NEGATIVE_COLOR if value < 0 else "#555555"
