import matplotlib.pyplot as plt
import numpy as np

from .R import (data_frame, pretty, ls, table, data_range, corr, first_true,
                unique_match)

from .bayes import SdPrior, NormalPrior, Ar1CoefficientPrior

from .data_table import to_data_table, to_data_frame, AutoClean

from .mcmc import suggest_burn

from .plots import (
    abline,
    barplot,
    boxplot,
    get_current_graphics_device,
    hist,
    histabunch,
    hosmer_lemeshow_plot,
    lines,
    plot,
    plot_dynamic_distribution,
    plot_many_ts,
    plot_grid_shape,
    plot_ts,
    points,
    )

from .probability import (
    dnorm, pnorm, qnorm, rnorm,
    dgamma, pgamma, qgamma, rgamma,
    rmarkov,
)

from .stats import density, sd
