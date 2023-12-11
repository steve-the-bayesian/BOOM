#!/usr/bin/env python3

import shutil
import os.path


def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.

        # Reduce the argument list by copying it starting from index 1.
        argv = argv[1:]
    return opts


def copy_many_files(filename_list, dest_dir, verbose=False):
    for fname in filename_list:
        dest = os.path.normpath(os.path.join(dest_dir, fname))
        if os.path.isdir(fname):
            if verbose:
                print(f"Doing nothing for {fname} because it is a directory.")
        else:
            if verbose:
                print('copying ', fname, ' to ', dest)
            target_directory = os.path.dirname(dest)
            if not os.path.exists(target_directory):
                os.makedirs(target_directory)
            shutil.copy(fname, dest)


if __name__ == '__main__':
    from sys import argv
    myargs = getopts(argv)
    program_name = argv[0]
    argv.remove(argv[0])
    dest_dir = argv.pop()
    copy_many_files(argv, dest_dir)
