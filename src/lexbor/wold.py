"""
Wrapper for handling data in wold package.
"""

import pickle
import random
from collections import Counter
from collections import OrderedDict

from lexbor import data
from lexbor import util


logger = util.get_logger(__name__)


class WoldDataset(data.LexibankDataset):
    def __init__(self, transform=None):
        """
        Load the data of the wold lexibank dataset.
        """
        super().__init__("wold", transform)

    def get_donor_table(self, language=None, form="Form", classification="Borrowed"):
        out = []
        for row in self.forms:
            if not language or row["Language"] == language:
                out.append(
                    [
                        row["ID"],
                        row[form],
                        row[classification],
                        row["donor_language"],
                        row["donor_description"],
                        row["donor_value"],
                        row["age_description"],
                        row["age_start_year"],
                        row["age_end_year"],
                    ]
                )

        return out


# =============================================================================
# Apply user function to tables from WOLD.
# =============================================================================


def apply_function_by_language(
    languages, form=None, function=None, donor_num=0, min_borrowed=0
):
    """

    Parameters
    ----------
    languages : str or [str] or 'all'
        Language name, list of language names, or 'all' for all languages.
    form : str, optional
        Form designation from ['Tokens', 'FormChars', 'ASJP', 'DOLGO', 'SCA'].
        The internal default is 'Tokens'.
    function : function
        User provided function applied to each language table.
        Initial arguments from function are language (str), form (str),
        and table ([str, [str], str]) corresponding to [id, form, loan_flag].
    donor_num : int, optional
        Whether borrowed words are selected only for donor, where 0 is no selection,
        1, 2, n selects the nth donor. The default is 0.
    min_borrowed: int, optional
        Minimum number of borrowed words in the table.

    Returns
    -------
    None.

    Notes
    -----
    See example examples/cross_validate_models_example.py

    """
    logger.debug(f"Apply function to languages {languages} and form {form}.")

    wolddb = get_wold_access()
    languages = check_wold_languages(wolddb, languages)

    for language in languages:
        table = get_native_donor_table(
            wolddb, language,
            form=form,
            classification="Borrowed",
            donor_num=donor_num
        )

        # Check for min_borrowed.
        num_borrowed = sum([row[2] for row in table])
        if num_borrowed < min_borrowed:
            logger.info(
                f"Table for {language} with donor {donor_num} "
                + f"has only {num_borrowed} borrowed words.  Skipped."
            )
        else:
            function(language, form, table)


# =============================================================================
# Get tables from WOLD via generator.
# =============================================================================

def language_table_gen(languages="all", form="Tokens",
                       donor_num=0, min_borrowed=0):
    """
    Get and present language tables one at a time in generator pattern.
    Option to select borrowed data for lead donor only.

    Parameters
    ----------
    languages : str or [str] or 'all'
        Language name, list of language names, or 'all' for all languages.
    form : str, optional
        Form designation from ['Tokens', 'FormChars', 'ASJP', 'DOLGO', 'SCA'].
        The internal default is 'Tokens'.
    donor_num: int, optional
        Whether borrowed words are selected only for donor, where 0 is no selection,
        1, 2, n selects the nth donor. The default is 0.
    min_borrowed: int, optional
        Minimum number of borrowed words in the table.

    Yields
    ------
    language : str
        Name of language.
    table : [str, [str], int]
        Table of [concept, [sound segments], borrowed_binary_flag].

    """

    logger.debug(f"Generator for {languages} languages.")

    wolddb = get_wold_access()
    languages = check_wold_languages(wolddb, languages)

    for language in languages:
        table = get_native_donor_table(
            wolddb, language,
            form=form,
            classification="Borrowed",
            donor_num=donor_num
        )

        # Check for min_borrowed.
        num_borrowed = sum([row[2] for row in table])
        if num_borrowed < min_borrowed:
            logger.info(
                f"Table for {language} with donor {donor_num} "
                + f"has only {num_borrowed} borrowed words.  Skipped."
            )
        else:
            yield language, table

# =============================================================================
# Get tables with donor number from WOLD via generator.
# =============================================================================


def language_table_gen_donor(languages="all", form="Tokens",
                             min_count=75, min_remainder=50):
    """
    Get and present language tables one at a time in generator pattern.
    Option to select borrowed data for lead donor only. Donor == 0 denotes inherited words.

    Parameters
    ----------
    languages : str or [str] or 'all'
        Language name, list of language names, or 'all' for all languages.
    form : str, optional
        Form designation from ['Tokens', 'FormChars', 'ASJP', 'DOLGO', 'SCA'].
        The internal default is 'Tokens'.
    min_count: int, optional
        Minimum number of borrowed words from a separate donor. The default is 75.
    min_remainder: int, optional
        Minimum number of remaining borrowed words in the table.

    Yields
    ------
    language : str
        Name of language.
    table : [str, [str], int]
        Table of [concept, [sound segments], donor_idx].
    sources : Counter of words for each donor.
    """

    logger.debug(f"Generator for {languages} languages.")

    wolddb = get_wold_access()
    languages = check_wold_languages(wolddb, languages)

    for language in languages:
        sources, language_table, table = get_language_table(
            wolddb, language=language, form=form,
            min_count=min_count, min_remainder=min_remainder
        )

        yield language, language_table, sources


# =============================================================================
# Language table access functions
# =============================================================================
def get_wold_access():
    def to_score(x):
        num = float(x.lstrip()[0])
        return (5 - num) / 4

    try:
        with open("wold.bin", "rb") as f:
            wolddb = pickle.load(f)
    except FileNotFoundError:
        wolddb = WoldDataset(
            transform={
                "Borrowed": lambda x, y, z: 1 if to_score(x["Borrowed"]) >= 0.9 else 0
            }
        )

        with open("wold.bin", "wb") as f:
            pickle.dump(wolddb, f)

    return wolddb


def check_wold_languages(wolddb, languages="all"):
    # Sort languages by orthographic order for the 'all' case.
    all_languages = sorted([language["Name"] for language in wolddb.languages.values()])
    # print(f"languages {all_languages}")
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

    logger.warning(
        "Language must be language name, list of languages, or keyword 'all'."
    )
    raise ValueError(f"Language list required, instead received {languages}.")


def get_donor(table, donor_num=0):
    if donor_num <= 0:
        return ""
    # Count and order the language donors.
    donors = Counter([row[3] for row in table if row[2] == 1]).most_common()
    if not donors or len(donors) == 0:
        return ""
    # Drop '' and 'Unidentified'.
    donors = [
        donor[0] for donor in donors if donor[0] != "" and donor[0] != "Unidentified"
    ]
    if len(donors) < donor_num:
        return ""
    return donors.pop(donor_num - 1)


def get_native_donor_table(
    wolddb, language, form="Tokens", classification="Borrowed", donor_num=0
):
    # Get table with borrowed words from donor_num only. Default to all donors.
    table = wolddb.get_donor_table(
        language=language, form=form, classification=classification
    )
    if donor_num == 0:
        table = [[row[0], row[1], row[2]] for row in table]
        return random.sample(table, len(table))

    # Select borrowed word for this donor.
    donor = get_donor(table, donor_num)
    # Select the rows containing this donor.
    # donor == '' selects all borrowed rows.
    donor_table = [
        [row[0], row[1], row[2]] for row in table if row[2] == 1 and donor in row[3]
    ]

    native_table = [[row[0], row[1], row[2]] for row in table if row[2] == 0]

    table_out = native_table + donor_table
    logger.info(
        f"Original {language} table size = {len(table)}, "
        + f"number from {donor} = {len(donor_table)}, "
        + f"number native = {len(native_table)}, "
        + f"table out size = {len(table_out)}."
    )

    # Return table of native words and borrowed words from lead donor.
    return random.sample(table_out, len(table_out))

#
# Functions to support multiple donor sources.
#


def get_language_sources(table, min_count=75, min_remainder=0):
    donors = Counter([row[3] for row in table if row[2] == 1])
    borrowed_count = sum(donors.values())
    inherited_count = len(table) - borrowed_count
    # Distinguish inherited from qualifying donors from remainder.
    # 0 is inherited, 1..<k is qualifying donors, k is remainder.
    remainder_count = 0
    for (key, count) in donors.most_common():
        if key == '' or key == 'Unidentified' or count < min_count:
            remainder_count += count
            del donors[key]
    sources = OrderedDict(donors)
    sources['inherited'] = inherited_count
    sources.move_to_end('inherited', last=False)

    if remainder_count < min_remainder and len(sources) > 1:
        # Merge remainder with last element.
        key, value = list(sources.items())[-1]
        sources[key] = sources[key] + remainder_count
    else:
        sources['remainder'] = remainder_count

    # Counter of donors from most to least frequent.
    return sources


def make_language_table(table, source_list):
    language_table = []
    for row in table:
        if row[2] == 0:
            # Inherited.
            language_table.append([row[0], row[1], row[2]])
        elif row[3] in source_list:  # Borrowed and in source_list.
            # Get source_list index.
            language_table.append([row[0], row[1], source_list.index(row[3], 1)])
        else:  # Borrowed and not in source_list.
            language_table.append([row[0], row[1], len(source_list) - 1])

    return language_table


# Test if remainder is too small to stand alone.
# If remainder < min_remainder then option to add to last donor

def get_language_table(
        wolddb, language, form="Tokens", min_count=75, min_remainder=0
):
    # Get table with structure [id, [segment], borrowed, donor]
    # id is arbitrary string or int id of token.
    # [segment] is list encoding of segments, where 'Tokens' results in IPA coded sound segments.
    # borrowed is status 0/1 where 0 is inherited and 1 is borrowed.
    # donor is string identifying donor language.
    #
    table = wolddb.get_donor_table(
        language=language, form=form, classification="Borrowed"
    )

    sources = get_language_sources(table, min_count=min_count, min_remainder=min_remainder)

    # Construct new table which has structure [id, [segment], source]
    # where source is from 'inherited', language, or 'remainder' sources.
    language_table = make_language_table(table, list(sources.keys()))

    return sources, language_table, table
