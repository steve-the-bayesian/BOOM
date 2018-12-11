COPTS = [
    "-I/usr/local/include",
    "-Iexternal/gtest/googletest-release-1.8.0/googletest/include",    
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
