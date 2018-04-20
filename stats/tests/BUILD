COPTS = [
    "-I/usr/local/include",
]

DEPS = [
    "//:boom",
    "//:boom_test_utils",
    "@gtest//:main",
]

cc_test(
    name = "logit_test",
    srcs = ["logit_test.cc"],
    copts = COPTS,
    deps = DEPS,
)

cc_test(
    name = "ecdf_test",
    srcs = ["ecdf_test.cc"],
    copts = COPTS,
    deps = DEPS,
)

cc_test(
    name = "resampler_test",
    srcs = ["resampler_test.cc"],
    copts = COPTS,
    deps = DEPS,
)
