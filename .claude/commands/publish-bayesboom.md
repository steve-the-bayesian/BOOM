---
description: Build the manylinux wheel and publish the current BayesBoom version to PyPI
---

Build a manylinux wheel for BayesBoom using the `Dockerfile` as a guide, then
extract the built package and upload any new versions to PyPI. Follow these
steps:

1. **Determine the current version.** Read `MAJOR`/`MINOR`/`PATCH` in
   `python_package/setup.py` to get the version being published (e.g. `0.2.6`).
   List `python_package/dist/` to see which wheels/sdists already exist locally.

2. **Build the manylinux wheel.** The host is arm64 but the base image
   (`quay.io/pypa/manylinux_2_28_x86_64`) is x86_64, so build with an explicit
   platform flag. Run in the repo root:

   ```
   docker build --platform linux/amd64 -t pyboom .
   ```

   This compiles the full C++ library via `./install/pyboom` and runs
   `auditwheel repair` inside the container. It takes a while (C++ compile under
   emulation) — run it in the background and monitor the build log for the
   `auditwheel repair` / `Fixed-up wheel written` milestone and for any errors.

3. **Extract the wheel** from the container into `python_package/dist/`:

   ```
   CID=$(docker create --platform linux/amd64 pyboom)
   docker cp "$CID:/output/bayesboom-<VERSION>-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" python_package/dist/
   docker rm "$CID"
   ```

   (Check the exact repaired filename in the build log's `Fixed-up wheel written
   to /output/...` line — the manylinux tag can vary.)

4. **Validate and publish.** From `python_package/`, run `twine check dist/*`.
   Then query PyPI (`https://pypi.org/simple/bayesboom/`) to see which files are
   already published, and upload only the genuinely new version's files with
   `--skip-existing` as a safety net:

   ```
   twine upload --skip-existing dist/bayesboom-<VERSION>*
   ```

   Credentials come from `~/.pypirc`.

5. **Confirm.** Verify the new files appear on the PyPI simple index
   (`https://pypi.org/simple/bayesboom/`) — it updates faster than the cached
   JSON API. Report which files were uploaded.

Only upload versions/files that are not already on PyPI. If the manylinux wheel
for the current version already exists both locally and on PyPI, say so instead
of rebuilding.
