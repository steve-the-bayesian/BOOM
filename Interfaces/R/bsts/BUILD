## Load the R build rules
load(
    "@com_grail_rules_r//R:defs.bzl",
    "r_pkg",
    "r_library",
    "r_unit_test",
    "r_pkg_test",
)

r_binary(
    name = "testdriver",
    src = "testdriver.R",
    deps = [
        "RUnit",
        "bsts",
    ],
)
