# Use the github archive for googletest / gtest, version 1.8
new_http_archive(
    name = "gtest",
    build_file = "gtest.BUILD",
    url = "https://github.com/google/googletest/archive/release-1.8.0.zip",
)

# Change master to the git tag you want.
http_archive(
    name = "com_grail_rules_r",
    strip_prefix = "rules_r-master",
    urls = ["https://github.com/grailbio/rules_r/archive/master.tar.gz"],
)

load("@com_grail_rules_r//R:dependencies.bzl", "r_rules_dependencies")

r_rules_dependencies()
