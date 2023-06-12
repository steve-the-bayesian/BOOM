# Use this base image from dockerhub.  This image is set up with the minimal
# linux environment to support manylinux distribution.
FROM quay.io/pypa/manylinux2010_x86_64

# Create the /output directory inside the container.
RUN mkdir /output

# Copy the contents of the BOOM directory to the container.
COPY . /src/BOOM

# Set the working directory for the container.
WORKDIR /src/BOOM

# Alias python3.8 on the container to 'python3'
RUN ln -sf /opt/python/cp38-cp38/bin/python /usr/local/bin/python3

# Run the install script.
RUN ./install/pyboom

# Grab the wheel from the 'dist' directory created by the install script and
# move it to the /output directory.
RUN cd python_package/dist && find . -name "*.whl" -print \
    && mv *.whl /output && (rm -f /usr/local/bin/python3 || true)

# The wheel in the output dirctory is specific to this platform.  Run the
#  'auditwheel' utility to produce the manylinux wheel from this
#  platform-specific wheel.
RUN auditwheel repair /output/BayesBoom*.whl -w /output
