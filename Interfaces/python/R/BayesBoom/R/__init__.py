import matplotlib.pyplot as plt
import numpy as np

from .R import (
    data_frame,
    pretty,
    ls,
    table,
    data_range,
    corr,
    first_true,
    paste,
    paste0,
    print_timestamp,
    recycle,
    remove_common_prefix,
    remove_common_suffix,
    unique_match,
)

from .bayes import (
    Ar1CoefficientPrior,
    DoubleModel,
    GaussianSuf,
    MvnPrior,
    MvnGivenSigma,
    NormalPrior,
    UniformPrior,
    WishartPrior,
    SdPrior,
)

from .density import Density

from .data_table import to_data_table, to_data_frame

from .autoclean import AutoClean

from .encoding import (
    EffectEncoder,
    IdentityEncoder,
    InteractionEncoder,
    DatasetEncoder,
)

from .graphics_device import get_current_graphics_device

from .mcmc import suggest_burn

from .plots import (
    abline,
    barplot,
    boxplot,
    compare_dynamic_distributions,
    hist,
    histabunch,
    hosmer_lemeshow_plot,
    lines,
    lty,
    mosaic_plot,
    plot,
    plot_dynamic_distribution,
    plot_many_ts,
    plot_grid_shape,
    plot_ts,
    points,
    time_series_boxplot,
    )

from .probability import (
    dnorm, pnorm, qnorm, rnorm,
    dgamma, pgamma, qgamma, rgamma,
    rmarkov,
)

from .stats import density, sd

from .test_utilities import delete_if_present

from .boom_py_utils import (
    is_all_numeric,
    is_iterable,
    to_boom_date,
    to_boom_vector,
    to_boom_matrix,
    to_boom_spd,
    to_numpy,
    to_pd_timestamp,
)
