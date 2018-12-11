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
