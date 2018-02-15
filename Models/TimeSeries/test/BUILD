COPTS = [
    "-I/usr/local/include",
    "-Iexternal/gtest/include",
]

# LIBS = [
#     "-L/usr/local/lib",
#     "-L/usr/lib",
#     "-llapack",
#     "-lblas",
#     "-lm",
# ]

cc_test(
    name = "ArmaModel_test",
    srcs = ["ArmaModel_test.cc"],
    copts = COPTS,
    #    linkopts = LIBS,
    deps = [
        "//:boom",
        "@gtest//:main",
    ],
)
