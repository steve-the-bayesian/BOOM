import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


_active_graphics_devices = {}
_current_graphics_device = None
_largest_graphics_device_number = 0


class GraphicsDevice(ABC):
    """
    Manages a plt.figure and a set of axes.
    """

    def __init__(self):
        """
        Creating a new graphics device updates the global set of devices
        """
        global _largest_graphics_device_number
        global _current_graphics_device
        global _active_graphics_devices
        self._number = _largest_graphics_device_number + 1
        _largest_graphics_device_number = self._number
        _active_graphics_devices[self._number] = self
        _current_graphics_device = self

        self._figure, self._axes = plt.subplots(1, 1)
        self._axes_cursor = None
        self._nrow = 1
        self._ncol = 1

    def __del__(self):
        plt.close(self._figure)
        global _active_graphics_devices
        del _active_graphics_devices[self._number]

    @property
    def next_axes(self):
        """
        The next set of axes on which a plot is to be drawn.  If the last set of
        available axes has been exhausted, the stored Figure is refreshed with
        new axes of the same dimension.
        """
        if hasattr(self._axes, "shape"):
            shape = self._axes.shape
            if min(shape) == 1:
                self._increment_1d_cursor()
            else:
                self._increment_2d_cursor()
        else:
            self._axes = self._figure.subplots(1, 1)
        return self.current_axes

    @property
    def current_axes(self):
        """
        The current set of axes.
        """
        if self._nrow > 1 or self._ncol > 1:
            return self._axes[self._axes_cursor]
        else:
            return self._axes

    @abstractmethod
    def draw_current_axes(self):
        """
        A hook to be called after the graphics device is updated.
        """

    def _create_subplots(self, nrow, ncol):
        """
        Set the graphics device to use multiple rows and/or columns of plots.
        """
        self._axes = self._figure.subplots(nrow, ncol)
        self._nrow = nrow
        self._ncol = ncol
        if min(nrow, ncol) > 1:
            self._axes_cursor = (0, 0)
        elif max(nrow, ncol) > 1:
            self._axes_cursor = 0
        else:
            self._axes_cursor = None

    def _increment_1d_cursor(self):
        """
        Increment the self._axes cursor when self._axes is a 1-d array.
        """
        dim = max(self._nrow, self._ncol)
        self._axes_cursor += 1
        if self._axes_cursor >= dim:
            self._axes_cursor = 0
            # We need nrow and ncol here because a 1-d axes might be a row, or
            # a column.
            self._axes = self._figure.subplots(self._nrow, self._ncol)

    def _increment_2d_cursor(self):
        """
        Increment self._axes_cursor when self._axes is two-dimensional.
        """
        i, j = self._axes_cursor
        j += 1
        if j >= self._axes.shape[1]:
            j = 0
            i += 1
            if i >= self._axes.shape[0]:
                self._axes = self._figure.subplots(
                    self._axes.shape[0], self._axes.shape[1])
                i, j = 0, 0
        self._axes_cursor = (i, j)


def dev_new():
    return InteractiveGraphicsDevice()


def dev_set(device_number: int):
    """
    Set the current graphics device to the device with the given number.

    """
    global _active_graphics_devices
    global _current_graphics_device
    device = _active_graphics_devices.get(device_number, None)
    if device is None:
        raise Exception(f"Graphics device {device_number} does note exist.")
    else:
        _current_graphics_device = device


class InteractiveGraphicsDevice(GraphicsDevice):
    """
    A private stack manages the set of active graphics devices.
    """

    def __init__(self):
        super().__init__()
        plt.show(block=False)
        plt.pause(.001)

    def draw_current_axes(self):
        """
        """
        plt.pause(.001)


class PdfGraphicsDevice(GraphicsDevice):
    """
    Destructor generates a PDF file that gets generated when the graphics
    device is deleted.
    """

    def __init__(self, filename, width=5, height=5):
        self._filename = filename

    def __del__(self):
        """
        Create the pdf file upon deletion.
        """
        if self._figure is not None:
            self._figure.save(self._filename)

    def draw_current_axes(self):
        """
        """
        pass


def get_current_graphics_device():
    """
    Returns the current graphics device, if one exists.  Otherwise create and
    return an interactive graphics device.
    """
    global _current_graphics_device
    if _current_graphics_device is not None:
        return _current_graphics_device
    else:
        return InteractiveGraphicsDevice()
