COPTS = [
    "-I/usr/local/include",
]

cc_test(
    name = "sparse_matrix_test",
    srcs = ["sparse_matrix_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:main",
    ],
)
