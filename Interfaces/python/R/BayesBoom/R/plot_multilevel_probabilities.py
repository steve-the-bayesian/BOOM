import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .plots import ensure_ax


def plot_multilevel_probabilities(prob_info,
                                  colors=None,
                                  logscale=True,
                                  ax=None,
                                  fig=None,
                                  **kwargs):
    """
    Args:
      prob_info: A list of pd.Series objects.  The index of each series element
        is a local taxonomy level.  The value is the conditional probability of
        choosing that level given a choice of the parent level.

        An example entry is
        food    .7
        steak   .6
        NYStrip .8

        In this example the probability of NYStrip is .7 * .6 * .8.  The
        probability of a non-food option somewhere else is .3, and the
        probability of a non-steak option given the food category is .4.

        The Series elements in prob_info can be of different lengths, either
        because the taxonomy from which they are drawn has different depths in
        different branches of the taxonomy tree, or because an observation is
        incompletely observed.

      logscale: Plot the probability contributions of each taxonomy level on the
        log scale.  This is the natural scale on which to see the individual
        contributions, but not to judge the final probability score of the
        result.
    
      colors: A color pallette to cycle through as different probability levels
        are encountered.

      ax:  The plt.Axes object on which to draw the plot.

      fig: The plt.Figure object on which to draw the plot.
    
      kwargs: Additional options will be passed to lower level plotting
        functions.

    Returns:
      The plt.Figure and plt.Axes objects containing the plot.
    """
    import BayesBoom.boom as boom
    
    fig, ax = ensure_ax(fig, ax)

    bar_levels = [x.index.tolist() for x in prob_info]

    taxonomy = boom.Taxonomy(bar_levels)
    colors = np.array(colors)
    
    for bar_number, probs in enumerate(prob_info):
        totals = np.cumsum(np.log(probs))
        lo = np.insert(totals[:-1], 0, 0.0)
        hi = totals
        
        if not logscale:
            lo = np.exp(lo)
            lo[0] = 0
            hi = np.exp(hi)

        bar_color_index = taxonomy.index(bar_levels[bar_number])
        bar_colors = colors[bar_color_index]
            
        x = np.full(len(probs), bar_number)
        ax.bar(x, 
               hi,
               # bottom=lo,
               bottom=0,
               width=1,
               color=bar_colors)

    if fig is not None:
        plt.show()

    return fig, ax
    

class BarInfo:
    """
    Organizes the information needed to plot the bars in
    plot_multilevel_probabilities.  A BarInfo object describes a taxonomy
    element and all its descendents in terms of

    * The label of the taxonomy element described by the bar.
    * The probability that taxonomy element occurs.
    * The number of sub-levels contained in that taxonomy level.

    The BarInfo object also owns a dict pointing to the BarInfo objects
    describing its children.  It does not know about its parent.
    """
    
    def __init__(self, label=None, prob=None):
        """
        Args:
          label: The name describing this taxonomy element.  Do not include the
            names of parents or ancestors.
          prob: The conditional probability (given that its parent occurs) that
            this level of the taxonomy occurs.

        If both label and prob and None then this BarInfo object represents the
        root of the taxonomy.
        """
        self._label = label
        self._count = 0
        self._prob = prob
        self._info = {}

    def add_bar(self, probs):
        """
        Args:
          probs: A pd.Series containing the conditional probabilities of the
            levels underneath this taxonomy level.
        """
        if probs.empty:
            return

        self._count += 1
        
        label = probs.index[0]
        prob = probs.iloc[0]
        info = self._info.get(label, BarInfo(label, prob))
        sub = probs.iloc[1:]
        if not sub.empty:
            info.add_bar(sub)
        self._info[label] = info

    @property
    def count(self):
        """
        The number of descendents below this taxonomy element.
        """
        return max(self._count, 1)

    @property
    def prob(self):
        return self._prob

    def keys(self):
        return self._info.keys()
    
    def plot(self,
             fig=None,
             ax=None,
             start=0,
             colors=["red", "blue", "green", "yellow", "pink", "orange", "black"],
             parent_bar_height=None,
             parent_bar_color=None,
             parent_level_index=None,
             **kwargs):
        """
        Args:
          fig, ax: The matplotlib Figure and Axes objects on whicht to draw the
            plot.  If ax is None then new Figure and Axes objects will be created.
        
          start:  The number of the bar where the plot should begin.
        
          colors: A sequence of colors used to distinguish bars of different
            levels.  If the number of colors is too small then colors will be
            recycled.

          parent_bar_height: The height (in terms of probability) of the parent
            bar containing this plot.  This is an implementation detail of the
            recursion generating the plot.  Calling code should pass 'None'.
        
          parent_bar_color: The color parent bar containing this plot.  This is
            an implementation detail of the recursion generating the plot.
            Calling code should pass 'None'.

          parent_level_index: The index of the parent level in its branch of the
            taxonomy.  Calling code at the top level should pass 'None'.
        
          **kwargs:  Additional named arguments are passed to Axes.bar().

        Returns:
          The Figure and Axes objects (as a (fix, ax) pair) on which the plot is
          drawn.
        """
        
        fig, ax = ensure_ax(fig, ax)

        # Plot the bar for this taxonomy element.
        if self._prob is not None:
            ax.bar(start,
                   self.prob,
                   width=self.count,
                   color=colors[0],
                   **kwargs)

        # Plot each of the bars for the children.
        if self._info:
            widths = [v.count for k, v in self._info.items()]
            starting_points = start + np.insert(np.cumsum(widths)[:-1], 0, 0)
            keys = self.keys()
            assert(len(keys) == len(starting_points))
            for i, k in enumerate(keys):
                self[k].plot(fig=fig,
                             ax=ax,
                             start=starting_points[i],
                             colors=colors,
                             **kwargs)
        
        return fig, ax

    def __repr__(self):
        ans = self._as_string(0)
        if ans.endswith("\n"):
            ans = ans[:-1]
        return ans

    def _describe(self):
        if self._label is None:
            return f"Top level node, with {len(self._info)} children"
        else:
            return f" [{self._label}]: {self._prob} with {len(self._info)} children"
        
    def _as_string(self, indent):
        ans = ' ' * indent + self._describe() + '\n'
        for label, info in self._info.items():
            ans += info._as_string(indent + 2)
        return ans

    def __getitem__(self, key):
        return self._info[key]

        
def plot_multilevel_probabilities(prob_info,
                                  colors=None,
                                  ax=None,
                                  fig=None,
                                  **kwargs):
    """
    Args:
      prob_info: A list of pd.Series objects.  The index of each series element
        is a local taxonomy level.  The value is the conditional probability of
        choosing that level given a choice of the parent level.

        An example entry is
        food    .7
        steak   .6
        NYStrip .8

        In this example the probability of NYStrip is .7 * .6 * .8.  The
        probability of a non-food option somewhere else is .3, and the
        probability of a non-steak option given the food category is .4.

        The Series elements in prob_info can be of different lengths, either
        because the taxonomy from which they are drawn has different depths in
        different branches of the taxonomy tree, or because an observation is
        incompletely observed.

      colors: A color pallette to cycle through as different probability levels
        are encountered.

      ax:  The plt.Axes object on which to draw the plot.

      fig: The plt.Figure object on which to draw the plot.
    
      kwargs: Additional options will be passed to lower level plotting
        functions.

    Returns:
      The plt.Figure and plt.Axes objects containing the plot.
    """
    
    fig, ax = ensure_ax(fig, ax)
    bar_info = BarInfo()
    for bar in prob_info:
        bar_info.add_bar(bar)
    bar_info.plot(fig, ax, **kwargs)
    return fig, ax
        
    
