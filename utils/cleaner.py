"""
Helpers to clean text
"""

import re


# matches non-alphanumeric, space, or sentence-ending punctuation
stripMatcher = re.compile(r'[^a-zA-Z\t\n\s_.?!:;/<>*&^%$#@()"~`+-]')
# matches any sequence of tabs, newlines, spaces, underscores, and dashes
spaceMatcher = re.compile(r'[\t\n\s_.?!:;/<>*&^%$#@()"~`+-]+')


def clean_text(rawString):
    """
    Cleans rawString by replacing spaceMatcher and tagMatcher with a single
    space, removing non-alpha chars, and lowercasing alpha chars
    """
    # replace stripMatcher with ""
    cleanedString = re.sub(stripMatcher, "", rawString)
    # replace spaceMatcher with " " and strip surround whitespace
    spacedString = re.sub(spaceMatcher, " ", cleanedString).strip()
    # lowercase the alpha chars that remain
    loweredString = spacedString.lower()
    return loweredString
