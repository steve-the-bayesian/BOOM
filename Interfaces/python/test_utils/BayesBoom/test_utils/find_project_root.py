import os
import pathlib


def find_project_root(dir=None):
    """
    Return the first enclosing directory containing a file named 'setup.py'
    """
    if dir is None:
        dir = os.path.curdir

    if os.path.exists(os.path.join(str(dir), "setup.py")):
        return dir
    else:
        current_path = pathlib.path(dir)
        parent = current_path.parent.absolute()
        if parent == current_path:
            return current_path
        else:
            return find_project_root(parent)
