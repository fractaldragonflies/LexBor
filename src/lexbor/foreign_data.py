"""
@author: John E. Miller

Created on June 4, 2021
Revised for lexbor on Sep 28, 2024

Function to get language tables from non-WOLD sources.

"""
import csv


def get_supplementary_data(filename):
    # Same temporary format as for cognate study.
    # words: [doculect, concept, tokens, borrowed]
    # doculect = language name in English
    # concept = concepticon_id
    # tokens = space delimited list of IPA segments
    # borrowed = True/False/Default=False
    # returns: [doculect, concepticon_id, [segments], borrowed]
    # All words are returned and borrowed is set to False for now.

    data = []
    with open(filename) as fl:
        rdr = csv.reader(fl, delimiter='\t', quotechar='"')
        _ = next(rdr)  # Ignore header.
        # hdr = next(rdr)
        # print(f'Header: {hdr}')

        for row in rdr:
            word = row[2]
            # Split based on space.
            segments = word.split(' ')
            borrowed = not row[3] or row[3] == 'False'
            data.append([row[0], row[1], segments, borrowed])

    return data


def get_lexbor_format_data(filename):
    # Same temporary format as for cognate study.
    # words: [concept, tokens, borrowed or donor]
    # concept = concepticon_id
    # tokens = space delimited list of IPA segments
    # borrowed = True/False/Default=False
    # donor = 0:inherited, 1+ index of donor
    # returns: [concepticon_id, [segments], borrowed or donor]
    # format [str, [str], int]
    # All words are returned and borrowed is set to False for now.

    data = []
    with open(filename) as fl:
        rdr = csv.reader(fl, delimiter='\t', quotechar='"')
        _ = next(rdr)  # Ignore header.

        for row in rdr:
            word = row[1]
            # Split based on space.
            segments = word.split(' ')
            data.append([row[0], segments, int(row[2])])

    return data


# if __name__ == "__main__":
#     # data = get_supplementary_data("../tables/Spanish.tsv")
#     data = get_pybor_format_data("../../tables/Shipibo-Konibo.pybor-format.tsv")
#     for item in data:
#         print(item)
