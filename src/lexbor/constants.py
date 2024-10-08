"""
Created on Thu Sep 26 19:14:01 2024

@author: johnmiller

Constants using Enum.
"""
from enum import Enum

# Enum for Math Constants
# class MathConstants(Enum):
#     PI = 3.14159
#     E = 2.71828
#
# To reference constant.
#     fmt = MathConstants.PI
#
#     if result == MathConstants.PI: ...
#
# To get constant value.
#     print(fmt.value)


# Enum for dataset format constants.
class DataFmtConst(Enum):
    STD = "[[id, [segmented_IPA], donor]]"
    SEG = "[[segmented_IPA]]"
    UNK = "Not STD or SEG format"

class DataBasisConst(Enum):
    ALL = "All combined"
    INH = "Inherited"
    BOR = "Borrowed"
    BOTH = "Both inherited and borrowed"

# =============================================================================
#     Utility functions returning constants
# =============================================================================
def is_list_str(ele):
    return (ele and isinstance(ele, list)
            and all(isinstance(s, str) for s in ele))

def detect_input_fmt(data:list):
    row = data[0]  # get initial row to detect dataset format.
    if len(row) == 3 and is_list_str(row[1]):
        fmt = DataFmtConst.STD
    elif is_list_str(row):
        fmt = DataFmtConst.SEG
    else:
        fmt = DataFmtConst.UNK
    return fmt
