# Use the github archive for googletest / gtest, version 1.8

new_http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.8.0.zip",
#    sha256 = "b58cb7547a28b2c718d1e38aee18a3659c9e3ff52440297e965f5edffe34b6d0",
    build_file = "gtest.BUILD",
#    strip_prefix = "googletest-release-1.8.0",
)


# Change master to the git tag you want.
http_archive(
    name = "com_grail_rules_r",
    strip_prefix = "rules_r-master",
    urls = ["https://github.com/grailbio/rules_r/archive/master.tar.gz"],
)

load("@com_grail_rules_r//R:dependencies.bzl", "r_rules_dependencies")

r_rules_dependencies()
