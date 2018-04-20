COPTS = [
    "-I/usr/local/include",
    "-Wno-sign-compare",
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

cc_test(
    name = "liu_west_particle_filter_test",
    srcs = ["liu_west_particle_filter_test.cc"],
    copts = COPTS,
    deps = [
        "//:boom",
        "//:boom_test_utils",
        "@gtest//:main",
    ],
)
