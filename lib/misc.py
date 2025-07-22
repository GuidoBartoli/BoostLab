"""Miscellaneous functions."""

import math
from datetime import datetime, timedelta
from decimal import Decimal
from time import time


def time2str(
    seconds: float, elapsed: bool = False, separator: str = ":", full: bool = False
) -> str:
    """Convert seconds to time string.

    :param seconds: seconds to convert
    :param elapsed: show elapsed time
    :param separator: separator character
    :param full: print full time string
    :return: time converted to string
    """
    if elapsed:
        seconds = time() - seconds
    if not full:
        if seconds < 1:
            return f"{int(seconds * 1000)} ms"
        if seconds < 60:
            return f"{seconds:.2f} s"
    date = datetime(1, 1, 1) + timedelta(seconds=seconds)
    if separator is None:
        return f"{date.hour:02d}h{date.minute:02d}m{date.second:02d}s"
    return f"{date.hour:02d}{separator}{date.minute:02d}{separator}{date.second:02d}"


def rngs2vals(ranges: str) -> list:
    """Convert list of ranges and single values into list (ex: '1,5:10,15' -> [1,5,6,7,8,9,10,15]).

    :param ranges: list of ranges or single values
    :return: single value list
    """
    # FIXME: Check if ranges contains only numbers and colons
    ranges = ranges.replace(" ", "")
    values = []
    for token in ranges.split(","):
        if not token:
            continue
        if ":" in token:
            limits = token.split(":")
            if len(limits) != 2:
                continue
            try:
                start, end = int(limits[0]), int(limits[1])
            except ValueError:
                continue
            if end <= start or start < 0 or end < 0:
                continue
            values.extend(range(start, end + 1))
        else:
            try:
                number = int(token)  # numeric values
                if number < 0:
                    continue  # discard negative values
                values.append(number)
            except ValueError:
                values.append(token)  # string values
    result = set()
    return [
        x for x in [str(x) for x in values] if x not in result and not result.add(x)
    ]


def shorten(array: list, length: int = 3) -> str:
    """Shorten a longer list of values.

    :param array: list of values
    :param length: maximum number of values to show
    :return: shortened list of values as string
    """
    return str(list(array[:length]) + ["..."]) if len(array) > length else str(array)


def lst2str(array: list, separator: str = " ") -> str:
    """Convert a list of values into a string.

    :param array: list of values
    :param separator: separator character
    :return: string with list of values
    """
    return separator.join([str(x) for x in array])


def humanize(
    number: int,
    precision: int = 0,
    drop_nulls: bool = True,
    prefixes: list = None,
    approx: bool = False,
) -> str:
    """Convert an integer number into human-readable format.

    :param number: number to convert
    :param precision: number of decimal places
    :param drop_nulls: drop trailing zeros
    :param prefixes: list of prefixes
    :param approx: approximate result
    :return: human-readable number
    """
    if prefixes is None:
        prefixes = []
    mill_names = ["", "k", "M", "B", "T", "P", "E", "Z", "Y"]
    if prefixes:
        mill_names = ["", *prefixes]
    number = float(number)
    mill_idx = max(
        0,
        min(
            len(mill_names) - 1,
            int(math.floor(0 if number == 0 else math.log10(abs(number)) / 3)),
        ),
    )
    result = "{:.{precision}f}".format(
        number / 10 ** (3 * mill_idx), precision=precision
    )
    if drop_nulls:
        result = Decimal(result)
        result = (
            result.quantize(Decimal(1))
            if result == result.to_integral()
            else result.normalize()
        )
    humanized = "{0}{dx}".format(result, dx=mill_names[mill_idx])
    if approx and humanized[-1] in mill_names:
        humanized = "~" + humanized
    return humanized


def round_list(array: list, precision: int = 4) -> list:
    """Round list of numbers.

    :param array: list of numbers
    :param precision: number of decimal places
    :return: rounded list of numbers
    """
    return [round(x, precision) for x in array]
