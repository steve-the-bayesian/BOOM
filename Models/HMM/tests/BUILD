COPTS = [
    "-I/usr/local/include",
]

cc_test(
    name = "hmm_test",
    srcs = ["hmm_test.cc"],
    copts = COPTS,
    data = ["fetal.lamb.data"],
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:main",
    ],
)
