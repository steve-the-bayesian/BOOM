load("@com_grail_rules_r//R:defs.bzl", "r_pkg", "r_library", "r_unit_test", "r_pkg_test")

package(default_visibility = ["//visibility:public"])

r_pkg(
    name = "BoomTestUtils",
    srcs = glob(
        ["**"],
        exclude = [],
    ),
    deps = [],
)

r_library(
    name = "library",
    pkgs = [":BoomTestUtils"],
    tags = ["manual"],
)
