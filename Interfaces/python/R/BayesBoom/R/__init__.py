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
    invert_order,
    order,
    paste,
    paste0,
    print_timestamp,
    recycle,
    remove_common_prefix,
    remove_common_suffix,
    unique_match,
    var,
    which,
)

from .bayes import (
    Ar1CoefficientPrior,
    BetaPrior,
    DoubleModel,
    GaussianSuf,
    MvnPrior,
    MvnGivenSigma,
    NormalPrior,
    SdPrior,
    UniformPrior,
    WishartPrior,
)

from .cbind import cbind

from .density import Density

from .data_table import to_data_table, to_data_frame

from .autoclean import AutoClean

from .encoding import (
    register_encoding_json_encoder,
    Encoder,
    MainEffectEncoder,
    MainEffectEncoderJsonEncoder,
    MainEffectEncoderJsonDecoder,
    EffectEncoder,
    OneHotEncoder,
    IdentityEncoder,
    InteractionEncoder,
    SuccessEncoder,
    SuccessEncoderJsonEncoder,
    SuccessEncoderJsonDecoder,
    DatasetEncoder,
    DatasetEncoderJsonEncoder,
    DatasetEncoderJsonDecoder,
)

from .graphics_device import get_current_graphics_device

from .mcmc import suggest_burn, report_progress

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
    pairs,
    plot,
    plot_dynamic_distribution,
    PlotDynamicDistribution,
    plot_many_ts,
    plot_grid_shape,
    plot_ts,
    points,
    time_series_boxplot,
    )

from .probability import (
    dmvn, rmvn,
    dnorm, pnorm, qnorm, rnorm,
    dgamma, pgamma, qgamma, rgamma,
    rbeta,
    rpois,
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
