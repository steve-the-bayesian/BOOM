cc_library(
    name = "test_utils",
    testonly = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.hpp"]),
    copts = ["-Wno-sign-compare"],
    visibility = [
        "//Models/StateSpace/StateModels/tests:__pkg__",
        "//Models/StateSpace/tests:__pkg__",
    ],
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@googletest//:gtest_main",
    ],
)
