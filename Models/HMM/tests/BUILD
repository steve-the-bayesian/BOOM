COPTS = [
    "-Wno-sign-compare",
]

COMMON_DEPS = [
        "//:boom",
        "//:boom_test_utils",
        "@googletest//:gtest_main",
]
    
cc_test(
    name = "hmm_test",
    srcs = ["hmm_test.cc"],
    copts = COPTS,
    data = ["fetal.lamb.data"],
    deps = COMMON_DEPS, 
    size = "small",
)

cc_test(
    name = "liu_west_particle_filter_test",
    srcs = ["liu_west_particle_filter_test.cc"],
    copts = COPTS,
    deps = COMMON_DEPS,
    size = "small",
)
