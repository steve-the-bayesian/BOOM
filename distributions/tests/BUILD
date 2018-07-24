COPTS = [
    "-I/usr/local/include",
]

cc_test(
    name = "extreme_value_test",
    srcs = ["extreme_value_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:main",
    ],
)

cc_test(
    name = "mvn_test",
    srcs = ["mvn_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:main",
    ],
)
