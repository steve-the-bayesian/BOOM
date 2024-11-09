import matplotlib.pyplot as plt
import numpy as np

from .pretty import pretty

from .ls import ls

from .assign_classes import assign_classes, ClassAssigner

from .base import (
    data_frame,
    table,
    data_range,
    corr,
    first_true,
    invert_order,
    order,
    paste,
    paste0,
    print_time_interval,
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
    GammaModel,
    GaussianSuf,
    MvnBase,
    MvnPrior,
    MvnGivenSigma,
    NormalPrior,
    RegSuf,
    ScottZellnerMvnPrior,
    SdPrior,
    UniformPrior,
    WishartPrior,
)

from .cbind import cbind

from .density import Density

from .data_table import to_data_table, to_data_frame

from .autoclean import AutoClean

from .empirical_distribution import NumericEmpiricalDistribution, ECDF

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
    MissingDummyEncoder,
    SuccessEncoder,
    SuccessEncoderJsonEncoder,
    SuccessEncoderJsonDecoder,
    DatasetEncoder,
    DatasetEncoderJsonEncoder,
    DatasetEncoderJsonDecoder,
)

from .frequency_distribution import FrequencyDistribution

from .graphics_device import get_current_graphics_device

from .lm import LinearModel, lm, AnovaTable

from .mcmc import suggest_burn, report_progress

from .pandas_json import (
    PdDataFrameJsonEncoder,
    PdDataFrameJsonDecoder,
    PdSeriesJsonEncoder,
    PdSeriesJsonDecoder,
    PdIndexJsonEncoder,
    PdIndexJsonDecoder,
)

from .plots import (
    abline,
    AddSegments,
    barplot,
    boxplot,
    BoxplotTrue,
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

from .stats import density, sd, mean, acf, kl_divergence

from .summary import (
    summary,
    UnivariateSummary,
    NumericSummary,
    CategoricalSummary,
    DateTimeSummary,
)

from .test_utilities import delete_if_present

from .boom_py_utils import (
    is_all_numeric,
    is_iterable,
    to_boom_array,
    to_boom_date,
    to_boom_vector,
    to_boom_matrix,
    to_boom_labelled_matrix,
    to_boom_spd,
    to_numpy,
    to_pd_dataframe,
    to_pd_timestamp,
)
