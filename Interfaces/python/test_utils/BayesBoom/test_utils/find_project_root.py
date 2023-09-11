import os
import pathlib


def find_project_root(current_dir=None):
    """
    Return the first enclosing directory containing a file named 'setup.py'
    """
    if current_dir is None:
        current_dir = os.path.curdir

    if os.path.exists(os.path.join(str(current_dir), "setup.py")):
        return current_dir
    else:
        current_path = pathlib.Path(current_dir)
        parent = current_path.parent.absolute()
        if parent == current_path:
            return current_path
        else:
            return find_project_root(parent)
