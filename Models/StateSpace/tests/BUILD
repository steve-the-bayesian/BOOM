COPTS = [
    "-I/usr/local/include",
    "-Wno-sign-compare",
]

filegroup(
    name = "testdata",
    data = ["airpassengers.txt"],
)

cc_test(
    name = "state_space_gaussian_model_test",
    srcs = ["state_space_gaussian_model_test.cc"],
    copts = COPTS,
    data = [":testdata"],
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:main",
    ],
)
