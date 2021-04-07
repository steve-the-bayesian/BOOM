FROM quay.io/pypa/manylinux1_x86_64
RUN mkdir /output
COPY . /src/BOOM
WORKDIR /src/BOOM

RUN ln -sf /opt/python/cp36-cp36m/bin/python /usr/local/bin/python3
RUN ./install/pyboom
RUN cd python_package/dist && find . -name "*.whl" -print \
    && mv *.whl /output && (rm -f /usr/local/bin/python3 || true)
