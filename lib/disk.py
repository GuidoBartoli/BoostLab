import contextlib
import csv
import os
import shutil

import numpy as np


def get_files(
    folder: str,
    sort: bool = False,
    fullpath: bool = False,
    extensions: list = None,
    recursive: bool = False,
) -> list:
    """Return the list of files found in the input folder.

    :param folder: input folder to scan
    :param sort: sort resulting list by name
    :param fullpath: append full path to filenames
    :param extensions: required file extension
    :param recursive: recursive folder walk
    :return: list of files found
    """
    files = []
    if extensions is not None:
        extensions = [e.lower() for e in extensions]
    with contextlib.suppress(OSError):
        items = os.listdir(folder)
        for i in items:
            j = os.path.join(folder, i)
            ext = os.path.splitext(i)[1].lower()[1:]
            if os.path.isfile(j) and (extensions is None or ext in extensions):
                files.append(os.path.abspath(j) if fullpath else i)
            elif recursive and os.path.isdir(j):
                files += get_files(j, sort, fullpath, extensions, recursive)
    return sorted(files) if sort else files


def get_folders(folder: str, sort: bool = False) -> list:
    """Return the list of folders found in the input folder.

    :param folder: input folder
    :param sort: sort list by name
    :return: list of found folder (empty if an error occurred)
    """
    try:
        folders = next(os.walk(folder))[1]
        return sorted(folders) if sort else folders
    except StopIteration:
        return []


def count_lines(filename: str) -> int:
    """Return the number of lines of a text file.

    :param filename: input file
    :return: number of text lines
    """
    with open(filename, "r") as file:
        return len(file.readlines())


def get_basename(filename: str) -> str:
    """Extract file basename.

    :param filename: input filename
    :return: file basename
    """
    return os.path.basename(filename)


def get_folder(filename: str) -> str:
    """Extract file folder.

    :param filename: input filename
    :return: file folder
    """
    return os.path.dirname(filename)


def count_files(folder: str, recursive: bool = False) -> int:
    """Return the number of files found in the input folder.

    :param folder: input folder
    :param recursive: recursive search
    :return: number of files found
    """
    if recursive:
        return sum(len(files) for _, _, files in os.walk(folder))
    files = [f for r, d, f in os.walk(folder)]
    return len([item for sublist in files for item in sublist])


def check_folder(folder: str) -> bool:
    """Check if a folder exists in the provided path.

    :param folder: folder to check
    :return: True if folder exists, False otherwise
    """
    return os.path.exists(folder) and os.path.isdir(folder)


def check_file(filename: str) -> bool:
    """Check if a file exists in the provided path.

    :param filename: file to check
    :return: True if file exists, False otherwise
    """
    return os.path.exists(filename) and os.path.isfile(filename)


def join_paths(*args) -> str:
    """Join multiple paths into a single path.

    :param args: paths to join
    :return: joined path
    """
    return os.path.join(*args)


def rename_item(old_name: str, new_name: str) -> None:
    """Rename a file or directory.

    :param old_name: old name
    :param new_name: new name
    """
    os.rename(old_name, new_name)


def contains_files(folder: str, files: list) -> bool:
    """Check if a folder contains files.

    :param folder: folder to check
    :param files: list of files to check
    :return: True if folder contains files, False otherwise
    """
    return all(os.path.exists(os.path.join(folder, f)) for f in files)


def file_size(filename: str, humanize: bool = False) -> int:
    """Compute the size in bytes of a file on disk.

    :param filename: file to analyze
    :return: number of bytes on disk
    """
    try:
        return os.path.getsize(filename)
    except OSError:
        return 0


def folder_size(folder: str, recursive: bool = False) -> int:
    """Compute the size in bytes of a folder on disk.

    :param folder: folder to analyze
    :param recursive: descend into subfolders
    :return: total folder bytes
    """
    try:
        if not recursive:
            size = sum(
                file_size(os.path.join(folder, f))
                for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f))
            )
        else:
            size = 0
            for root, _, filenames in os.walk(folder):
                for filename in filenames:
                    size += file_size(os.path.join(root, filename))
        return size
    except OSError:
        return -1


def clean_folder(folder: str, recursive: bool = False):
    """Remove all items contained in a folder.

    :param folder: folder to clean
    :param recursive: descend into subfolders
    """
    with contextlib.suppress(OSError):
        for i in os.listdir(folder):
            item = os.path.join(folder, i)
            if os.path.isfile(item):
                os.remove(item)
            elif recursive and os.path.isdir(item):
                shutil.rmtree(item, ignore_errors=True)


def init_folder(folder: str, clean: bool = True):
    """Prepare an empty folder for output.

    :param folder: folder to be initialized
    :param clean: remove existing files
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    elif clean:
        clean_folder(folder, recursive=True)


def human_size(total: int, binary: bool = False) -> str:
    """Convert total bytes into human-readable format.

    :param total: number of bytes
    :param binary: use binary suffixes
    :return: human-readable string
    """
    units = ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]
    if binary:
        units = [f"{unit}i" for unit in units]
        factor = 1024.0
    else:
        factor = 1000.0
    suffix = "B"
    for unit in units:
        if abs(total) < factor:
            return f"{total:3.1f} {unit}{suffix}"
        total /= factor
    return f"{total:.1f} {units[-1]}{suffix}"


def get_ext(filename: str) -> str:
    """Extract file extension.

    :param filename: input filename
    :return: file extension
    """
    return os.path.splitext(filename)[1][1:].lower()


def append_ext(filename: str, extension: str) -> str:
    """Append specified extension (if not present) to filename.

    :param filename: file to check
    :param extension: required extension
    :return: filename with extension appended
    """
    if not extension.startswith("."):
        extension = f".{extension}"
    if not filename.endswith(extension):
        filename += extension
    return filename


def remove_ext(filename: str) -> str:
    """Remove extension from file.

    :param filename: filename with extension
    :return: filename with extension removed
    """
    return os.path.splitext(filename)[0]


def save_csv(
    filename: str,
    matrix: np.ndarray,
    headers: list = None,
    append: bool = False,
    sep: str = ",",
):
    """Save matrix to CSV file.

    :param filename: CSV filename
    :param matrix: matrix to save
    :param headers: first row names
    :param append: append data
    :param sep: value separator
    """
    with open(filename, "a" if append else "w") as file:
        if headers is not None:
            for i, title in enumerate(headers):
                file.write(str(title))
                if i < len(headers) - 1:
                    file.write(",")
            file.write("\n")
        if isinstance(matrix, np.ndarray) and len(matrix.shape) == 1:
            matrix = np.array([matrix])
        for row in matrix:
            for i, item in enumerate(row):
                file.write(str(item))
                if i < len(row) - 1:
                    file.write(sep)
            file.write("\n")


def load_csv(
    filename: str, maxrow: int = None, maxcol: int = None, skip: int = 1
) -> np.ndarray:
    """Load data from CSV file.

    :param filename: CSV filename
    :param maxrow: row limit
    :param maxcol: column limit
    :param skip: row skip
    """
    matrix = []
    cols = 0
    with open(filename, "r") as csvfile:
        for i, row in enumerate(csv.reader(csvfile)):
            if i % skip != 0:
                continue
            if i == 0:
                cols = len(row)
            elif len(row) != cols:
                continue
            newrow = []
            valid = True
            for j, value in enumerate(row):
                try:
                    newrow.append(float(value))
                    if maxcol is not None and j == maxcol:
                        break
                except ValueError:
                    if value != "":
                        valid = False
                        break
            if valid:
                matrix.append(newrow)
                if maxrow is not None and i == maxrow:
                    break
    return np.asarray(matrix)


def save_strings(filename: str, strings: list, append: bool = False):
    """Save a list of strings to file.

    :param filename: file to save
    :param strings: list of strings
    :param append: append mode
    """
    with open(filename, "a" if append else "w") as file:
        file.write("\n".join(strings))


def load_strings(filename: str) -> list:
    """Load a list of strings from file.

    :param filename: file to load
    :return: list of strings
    """
    with open(filename) as file:
        return file.read().splitlines()
