cc_library(
    name = "test_utils",
    testonly = True,
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.hpp"]),
    copts = ["-I/usr/local/include"],
    visibility = [
        "//Models/StateSpace/StateModels/tests:__pkg__",
        "//Models/StateSpace/tests:__pkg__",
    ],
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:gtest_main",
    ],
)
