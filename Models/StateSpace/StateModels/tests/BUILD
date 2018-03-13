COPTS = [
    "-I/usr/local/include",
]

cc_test(
    name = "Holiday_test",
    srcs = ["Holiday_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "@gtest//:main",
    ],
)

cc_test(
    name = "RegressionHolidayStateModel_test",
    srcs = ["RegressionHolidayStateModel_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "@gtest//:main",
    ],
)

cc_test(
    name = "HierarchicalRegressionHolidayStateModel_test",
    srcs = ["HierarchicalRegressionHolidayStateModel_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "@gtest//:main",
    ],
)

cc_test(
    name = "HolidayMcmc_test",
    srcs = ["HolidayMcmc_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "@gtest//:main",
    ],
)

cc_test(
    name = "HolidayMcmcExample",
    srcs = ["HolidayMcmcExample.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
    ],
)
