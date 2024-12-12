COPTS = [
    "-Wno-sign-compare",
]

COMMON_DEPS = [
        ":mv_statespace_test_framework",
        "//:boom",
        "//:boom_test_utils",
        "@googletest//:gtest_main",
]

SANITIZERS = [
    #        "-fsanitize=address",
    #        "-fsanitize=undefined",
    #        "-fsanitize=vptr",
    #        "-fsanitize=thread"
]

cc_library(
    name = "mv_statespace_test_framework",
    hdrs = [
    	 "mv_framework.hpp",
    	 "student_regression_framework.hpp",
	 ],
    visibility = ["//visibility:public"],
    deps = ["//:boom"],
)

cc_test(
    name = "multivariate_state_space_regression_test",
    size = "small",
    srcs = [
        "multivariate_state_space_regression_test.cc",
    ],
    copts = COPTS,
    deps = COMMON_DEPS, 
)

cc_test(
    name = "student_mvss_regression_test",
    size = "small",
    srcs = [
        "student_mvss_regression_test.cc",
    ],
    copts = COPTS,
    deps = COMMON_DEPS, 
)
