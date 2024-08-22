""" Created on Oct 12, 2021
@author John E. Miller

Wrapper for handling wold package with access to borrowed score and donor data.

"""

import pickle
import pybor.data


class WoldPlusDataset(data.LexibankDataset):
    def __init__(self, transform=None):
        """
        Load the data of the wold lexibank dataset.
        """
        super().__init__("wold", transform)

    def get_donor_table(self, language=None, form="Tokens", classification="Borrowed"):
        out = []
        for row in self.forms:
            if not language or row["Language"] == language:
                out.append(
                    [
                        # row["ID"],
                        row["Concepticon_GLOSS"],
                        row["Concepticon_ID"],
                        row[form],
                        row[classification],
                        row["Borrowed_score"],
                        row["donor_language"],
                        # row["donor_description"],
                        row["donor_value"],
                        # row["age_description"],
                        # row["age_start_year"],
                        # row["age_end_year"],
                    ]
                )

        return out

    def get_flat_table(self, languages=None):
        # First pass just to verify what works.
        out = []

        for row in self.forms:
            if not languages or row["Language"] in languages:
                concept = row["Concepticon_GLOSS"] if row["Concepticon_GLOSS"] else row["Concept"]
                if concept.startswith('the '):
                    concept = concept[len('the '):]
                if concept.startswith('to '):
                    concept = concept[len('to '):]
                if concept.startswith('a '):
                    concept = concept[len('a '):]
                out.append(
                    [
                        # row["ID"],
                        # row["Language_ID"],
                        # row["Family"],
                        row["Language"],
                        self.languages[row["Language_ID"]]["Family"],
                        # row["Concept"],
                        # row["Concepticon_GLOSS"],
                        concept,
                        row["Concepticon_ID"],
                        row["Value"],
                        row["Form"],
                        ' '.join(row["Tokens"]),
                        # processed tokens in format of space delimited string.
                        # row["Segments"],
                        # row["Graphemes"],
                        row["Borrowed"],
                        row["Borrowed_score"],
                        row["donor_language"],
                        # row["donor_description"],
                        row["donor_value"],
                        # row["age_description"],
                        # row["age_start_year"],
                        # row["age_end_year"],
                    ]
                )

        return out


# =============================================================================
# Language table access functions
# =============================================================================
def get_wold_access():
    def to_score(x):
        num = float(x.lstrip()[0])
        return (5 - num) / 4

    try:
        with open("woldplus.bin", "rb") as f:
            wolddb = pickle.load(f)
    except FileNotFoundError:
        wolddb = WoldPlusDataset(
            transform={
                "Borrowed_score": lambda x, y, z: to_score(x["Borrowed"]),
                "Borrowed": lambda x, y, z: 1 if to_score(x["Borrowed"]) >= 0.9 else 0,
            }
        )

        with open("woldplus.bin", "wb") as f:
            pickle.dump(wolddb, f)

    return wolddb


def check_wold_languages(wolddb, languages="all"):

    all_languages = sorted([language["Name"] for language in wolddb.languages.values()])

    print(languages)
    if "all" in languages:
        return all_languages

    if isinstance(languages, str):
        languages = [languages]

    if isinstance(languages, list):
        for language in languages:
            if language not in all_languages:
                raise ValueError(f"Language {language} not in Wold.")
        # Don't sort if user provided list.
        return languages  # Checked as valid.

    raise ValueError(f"Language list required, instead received {languages}.")
