# Use this base image from dockerhub.  This image is set up with the minimal
# linux environment to support manylinux distribution.
# FROM quay.io/pypa/manylinux2010_x86_64
FROM quay.io/pypa/manylinux2014_x86_64

# Create the /output directory inside the container.
RUN mkdir /output

# Copy the contents of the BOOM directory to the container.
COPY . /src/BOOM

# Set the working directory for the container.
WORKDIR /src/BOOM

# Alias python3.9 on the container to 'python3'
# RUN ln -sf /opt/python/cp39-cp39/bin/python /usr/local/bin/python3
RUN rm -f /usr/bin/python
RUN ln -sf /opt/python/cp310-cp310/bin/python3 /usr/bin/python
RUN ln -sf /opt/python/cp310-cp310/bin/pip3 /usr/bin/pip

# Code that was written to use /usr/bin/env shebangs doesn't work anymore.  The
# solution is to hard-wire the python interpreter we want to use.
RUN sed -i 's|usr/bin/env python3|usr/bin/python|' install/install_headers.py

# Run the install script.
RUN ./install/pyboom

# Grab the wheel from the 'dist' directory created by the install script and
# move it to the /output directory.
RUN cd python_package/dist && find . -name "*.whl" -print \
    && mv *.whl /output && (rm -f /usr/bin/python || true)

# The wheel in the output dirctory is specific to this platform.  Run the
#  'auditwheel' utility to produce the manylinux wheel from this
#  platform-specific wheel.
RUN auditwheel repair /output/BayesBoom*.whl -w /output



# If the docker daemon is running, you can type
#     docker pull quay.io/pypa/manylinux2014_x86_64
# to pull the image from dockerhub to the local machine.
#
# Then type
#     docker run -i -t quay.io/pypa/manylinux2014_x86_64 /bin/bash
# to start bash and run inside the container interactively.
#
# Once you're in, you can identify the relevant versions of python3 and pip and
# adjust them above.

# Once the job is done run
#      docker run -v /tmp:/export -i -t pyboom /bin/bash
#      cd /output
#      mv BayesBoom-0.1.14-cp310-cp310-*.whl /export
#      exit
# Then the wheels will be in /tmp
