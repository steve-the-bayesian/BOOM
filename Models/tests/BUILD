COPTS = [
    "-I/usr/local/include",
    "-Iexternal/gtest/googletest-release-1.8.0/googletest/include",
    "-Wno-sign-compare",
]

cc_test(
    name = "gaussian_test",
    srcs = ["gaussian_test.cc"],
    copts = COPTS,
    includes = ["@gtest"],
    deps = [
	"//:boom",
	"//:boom_test_utils",
	"@gtest//:main",
    ],
)

cc_test(
    name = "independent_mvn_test",
    srcs = ["independent_mvn_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:main",
    ],
)

cc_test(
    name = "matrix_normal_test",
    srcs = ["matrix_normal_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:main",
    ],
)

cc_test(
    name = "spddata_test",
    srcs = ["spddata_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:main",
    ],
)
