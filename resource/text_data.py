"""
Module with example data for the project.

The data is stored as list of tuples, where each tuple contains:
- audio_file (path to the audio file in the same directory)
- transcription (str)
- context (list of strings)
"""

TRAIN_DATA: list[tuple[str, str, list[str]]] = [
    [
        "herr_kalinowski_ein_glas_wasser_getrunken.mp3",
        "herr kalinowski hat einen glas wasser getrunken",
        ["herr kalinowski", "frau berbel"],
    ],
    [
        "herr_kalinowski_ein_glas_wasser_getrunken.mp3",
        "herr calinovski hat einen glas wasser getrunken",
        ["herr calinovski", "Fachwort"],
    ],
]

TEST_DATA: list[tuple[str, str, list[str]]] = [
    [
        "herr_kalinowski_ein_glas_wasser_getrunken.mp3",
        "herr kalinowski hat einen glas wasser getrunken",
        ["herr kalinowski", "frau berbel"],
    ],
    [
        "herr_kalinowski_ein_glas_wasser_getrunken.mp3",
        "herr calinovski hat einen glas wasser getrunken",
        ["herr calinovski", "Fachwort"],
    ],
    [
        "herr_kalinowski_ein_glas_wasser_getrunken.mp3",
        "herr kalinowski hat einen glas wasser getrunken",
        ["herr calinovski", "frau berbel"],
    ],
]

__all__ = ["TRAIN_DATA", "TEST_DATA"]
