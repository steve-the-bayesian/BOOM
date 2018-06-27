cc_library(
    name = "main",
    srcs = ["googletest-release-1.8.0/googletest/src/gtest_main.cc"],
    hdrs = glob([
        "googletest-release-1.8.0/googletest/include/**/*.h",
        "googletest-release-1.8.0/googletest/**/*.h",
    ]),
    copts = [
        "-Iexternal/gtest/googletest-release-1.8.0/googletest/include",
        "-Iexternal/gtest/googletest-release-1.8.0/googletest",
    ],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
    deps = [":gtest"],
)

# The Google test library, excluding the main function.  Use this for writing
# test utilities that depend on gtest macros.

cc_library(
    name = "gtest",
    srcs = glob(
        [
            "googletest-release-1.8.0/googletest/src/*.cc",
            "googletest-release-1.8.0/googletest/src/*.h",
        ],
        exclude =
            [
                "googletest-release-1.8.0/googletest/src/gtest_main.cc",
                "googletest-release-1.8.0/googletest/src/gtest-all.cc",
            ],
    ),
    hdrs = glob([
        "googletest-release-1.8.0/googletest/include/**/*.h",
        "googletest-release-1.8.0/googletest/**/*.h",
    ]),
    copts = [
        "-Iexternal/gtest/googletest-release-1.8.0/googletest/include",
        "-Iexternal/gtest/googletest-release-1.8.0/googletest",
    ],
    linkopts =
        ["-pthread"],
    visibility = ["//visibility:public"],
)
