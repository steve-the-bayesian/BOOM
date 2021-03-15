import os


def delete_if_present(fname):
    """
    Delete file 'fname' if it exists in the filesystem.

    Args:
      fname:  A string or similar object naming the path to a file.
    """
    if os.path.exists(fname):
        os.remove(fname)
