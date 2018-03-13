
## Use the github archive for googletest / gtest, version 1.8
new_http_archive(
    name = "gtest",
    url = "https://github.com/google/googletest/archive/release-1.8.0.zip",
    build_file = "gtest.BUILD",
)

## Grab bazel rules for R
## See https://github.com/grailbio/rules_r
http_archive(
    name = "com_grail_rules_r",
    strip_prefix = "rules_r-0.3.4",
    urls = ["https://github.com/grailbio/rules_r/archive/0.3.4.tar.gz"],
)
