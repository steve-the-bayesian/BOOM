import os

# This file contains utilities used to implement unit tests.  For unit tests
# verifying the correctness of utils.py, see test_utils.py.


def delete_if_present(fname):
    """
    Delete file 'fname' if it exists in the filesystem.

    Args:
      fname:  A string or similar object naming the path to a file.
    """
    if os.path.exists(fname):
        os.remove(fname)
