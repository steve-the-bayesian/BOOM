from setuptools import setup, Extension, find_packages, find_namespace_packages
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
from glob import glob
import distutils.ccompiler

# Bump the major version when making backwards incompatible changes.
MAJOR = 0

# Bump the minor version when adding backwards compatible features.
MINOR = 0

# Bump the patch version when making bug fixes.
PATCH = 2

__version__ = f'{MAJOR}.{MINOR}.{PATCH}'


# Note that this setup.py is somewhat nonstandard.  In the main BOOM repository
# stored on github, setup.py and the pybind11 bindings are kept in
# .../Interfaces/python/BayesBoom/... For setup.py to work the C++ code must be
# in a subdirectory below setup.py.
#
# A script in the top level of the BOOM project copies the BOOM source code
# into a build directory in a way that will make setup.py happy.  This file is
# intended to be run by that build script, and not directly from the
# repository.

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


BOOM_DIR = "BayesBoom/boom"
BOOM_DIR += "/"
boom_headers = glob(BOOM_DIR + "*.hpp")

distributions_sources = glob(BOOM_DIR + "distributions/*.cpp")
distributions_headers = (
    [BOOM_DIR + "distributions.hpp"]
    + glob(BOOM_DIR + "distributions/*.hpp")
    )

linalg_sources = glob(BOOM_DIR + "LinAlg/*.cpp")
linalg_headers = glob(BOOM_DIR + "LinAlg/*.hpp")

math_sources = glob(BOOM_DIR + "math/*.cpp")
math_sources += glob(BOOM_DIR + "math/cephes/*.cpp")

numopt_sources = glob(BOOM_DIR + "numopt/*.cpp")
numopt_headers = [BOOM_DIR + "numopt.hpp"] + glob(BOOM_DIR + "numopt/*.hpp")

rmath_sources = glob(BOOM_DIR + "Bmath/*.cpp")
rmath_headers = glob(BOOM_DIR + "Bmath/*.hpp")

samplers_sources = glob(BOOM_DIR + "Samplers/*.cpp")
samplers_sources += [BOOM_DIR + "Samplers/Gilks/arms.cpp"]
samplers_headers = glob(BOOM_DIR + "Samplers/*.hpp")

stats_sources = glob(BOOM_DIR + "stats/*.cpp")
stats_headers = glob(BOOM_DIR + "stats/*.hpp")

targetfun_sources = glob(BOOM_DIR + "TargetFun/*.cpp")
targetfun_headers = glob(BOOM_DIR + "TargetFun/*.hpp")

utils_sources = glob(BOOM_DIR + "cpputil/*.cpp")
utils_headers = glob(BOOM_DIR + "cpputil/*.hpp")

models_sources = (
    glob(BOOM_DIR + "Models/*.cpp")
    + glob(BOOM_DIR + "Models/PosteriorSamplers/*.cpp")
    + glob(BOOM_DIR + "Models/Policies/*.cpp"))
models_headers = (
    glob(BOOM_DIR + "Models/*.hpp")
    + glob(BOOM_DIR + "Models/Policies/*.hpp")
    + glob(BOOM_DIR + "Models/PosteriorSamplers/*.hpp"))

# Specific model classes to be added later, glm's hmm's, etc.
bart_sources = (
    glob(BOOM_DIR + "Models/Bart/*.cpp")
    + glob(BOOM_DIR + "Models/Bart/PosteriorSamplers/*.cpp")
    )
bart_headers = (
    glob(BOOM_DIR + "Models/Bart/*.hpp")
    + glob(BOOM_DIR + "Models/Bart/PosteriorSamplers/*.hpp")
    )

glm_sources = (
    glob(BOOM_DIR + "Models/Glm/*.cpp")
    + glob(BOOM_DIR + "Models/Glm/PosteriorSamplers/*.cpp")
    )
glm_headers = (
    glob(BOOM_DIR + "Models/Glm/*.hpp")
    + glob(BOOM_DIR + "Models/Glm/PosteriorSamplers/*.hpp")
    )

hmm_sources = (
    glob(BOOM_DIR + "Models/HMM/*.cpp")
    + glob(BOOM_DIR + "Models/HMM/Clickstream/*.cpp")
    + glob(BOOM_DIR + "Models/HMM/Clickstream/PosteriorSamplers/*.cpp")
    + glob(BOOM_DIR + "Models/HMM/PosteriorSamplers/*.cpp")
    )
hmm_headers = (
    glob(BOOM_DIR + "Models/HMM/*.hpp")
    + glob(BOOM_DIR + "Models/HMM/Clickstream/*.hpp")
    + glob(BOOM_DIR + "Models/HMM/Clickstream/PosteriorSamplers/*.hpp")
    + glob(BOOM_DIR + "Models/HMM/PosteriorSamplers/*.hpp")
    )

hierarchical_sources = (
    glob(BOOM_DIR + "Models/Hierarchical/*.cpp")
    + glob(BOOM_DIR + "Models/Hierarchical/PosteriorSamplers/*.cpp")
    )
hierarchical_headers = (
    glob(BOOM_DIR + "Models/Hierarchical/*.hpp")
    + glob(BOOM_DIR + "Models/Hierarchical/PosteriorSamplers/*.hpp")
    )

impute_sources = (
    glob(BOOM_DIR + "Models/Impute/*.cpp")
    )
impute_headers = (
    glob(BOOM_DIR + "Models/Impute/*.hpp")
    )

irt_sources = (
    glob(BOOM_DIR + "Models/IRT/*.cpp")
    + glob(BOOM_DIR + "Models/IRT/PosteriorSamplers/*.cpp")
    )
irt_headers = (
    glob(BOOM_DIR + "Models/IRT/*.hpp")
    + glob(BOOM_DIR + "Models/IRT/PosteriorSamplers/*.hpp")
    )

mixture_sources = (
    glob(BOOM_DIR + "Models/Mixtures/*.cpp")
    + glob(BOOM_DIR + "Models/Mixtures/PosteriorSamplers/*.cpp")
    )
mixture_headers = (
    glob(BOOM_DIR + "Models/Mixtures/*.hpp")
    + glob(BOOM_DIR + "Models/Mixtures/PosteriorSamplers/*.hpp")
    )

nnet_sources = (
    glob(BOOM_DIR + "Models/Nnet/*.cpp")
    + glob(BOOM_DIR + "Models/Nnet/PosteriorSamplers/*.cpp")
    )
nnet_headers = (
    glob(BOOM_DIR + "Models/Nnet/*.hpp")
    + glob(BOOM_DIR + "Models/Nnet/PosteriorSamplers/*.hpp")
    )

point_process_sources = (
    glob(BOOM_DIR + "Models/PointProcess/*.cpp")
    + glob(BOOM_DIR + "Models/PointProcess/PosteriorSamplers/*.cpp")
    )
point_process_headers = (
    glob(BOOM_DIR + "Models/PointProcess/*.hpp")
    + glob(BOOM_DIR + "Models/PointProcess/PosteriorSamplers/*.hpp")
    )

state_space_sources = (
    glob(BOOM_DIR + "Models/StateSpace/*.cpp")
    + glob(BOOM_DIR + "Models/StateSpace/Filters/*.cpp")
    + glob(BOOM_DIR + "Models/StateSpace/PosteriorSamplers/*.cpp")
    + glob(BOOM_DIR + "Models/StateSpace/StateModels/*.cpp")
)
state_space_headers = (
    glob(BOOM_DIR + "Models/StateSpace/*.hpp")
    + glob(BOOM_DIR + "Models/StateSpace/Filters/*.hpp")
    + glob(BOOM_DIR + "Models/StateSpace/PosteriorSamplers/*.hpp")
    + glob(BOOM_DIR + "Models/StateSpace/StateModels/*.hpp")
)

time_series_sources = (
    glob(BOOM_DIR + "Models/TimeSeries/*.cpp")
    + glob(BOOM_DIR + "Models/TimeSeries/PosteriorSamplers/*.cpp")
    )
time_series_headers = (
    glob(BOOM_DIR + "Models/TimeSeries/*.hpp")
    + glob(BOOM_DIR + "Models/TimeSeries/PosteriorSamplers/*.hpp")
    )

boom_library_sources = (
    distributions_sources
    + linalg_sources
    + math_sources
    + numopt_sources
    + rmath_sources
    + samplers_sources
    + stats_sources
    + targetfun_sources
    + utils_sources
    + models_sources
    + bart_sources
    + glm_sources
    + hmm_sources
    + hierarchical_sources
    + impute_sources
    + irt_sources
    + mixture_sources
    + nnet_sources
    + point_process_sources
    + state_space_sources
    + time_series_sources
)

boom_extension_sources = (
    [BOOM_DIR + "pybind11/module.cpp"]
    + glob(BOOM_DIR + "pybind11/Models/*.cpp")
    + glob(BOOM_DIR + "pybind11/Models/Glm/*.cpp")
    + glob(BOOM_DIR + "pybind11/Models/Impute/*.cpp")
    + glob(BOOM_DIR + "pybind11/Models/StateSpace/*.cpp")
    + glob(BOOM_DIR + "pybind11/Models/StateSpace/StateModels/*.cpp")
    + glob(BOOM_DIR + "pybind11/Models/TimeSeries/*.cpp")
    + glob(BOOM_DIR + "pybind11/LinAlg/*.cpp")
    + glob(BOOM_DIR + "pybind11/stats/*.cpp")
    + glob(BOOM_DIR + "pybind11/distributions/*.cpp")
)

boom_sources = boom_extension_sources + boom_library_sources


# ---------------------------------------------------------------------------
# From
# https://stackoverflow.com/questions/11013851/speeding-up-build-process-with-distutils
# monkey-patch for parallel compilation
def parallelCCompile(self, sources, output_dir=None, macros=None,
                     include_dirs=None, debug=0, extra_preargs=None,
                     extra_postargs=None, depends=None):

    # those lines are copied from distutils.ccompiler.CCompiler directly
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs)

    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
    # parallel code
    N = 16  # number of parallel compilations
    import multiprocessing.pool

    def _single_compile(obj):
        try:
            src, ext = build[obj]
        except KeyError:
            return
        self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

    # convert to list, imap is evaluated on-demand
    list(multiprocessing.pool.ThreadPool(N).imap(_single_compile, objects))
    return objects


distutils.ccompiler.CCompiler.compile = parallelCCompile
# End of parallel compile "monkey patch"
# ---------------------------------------------------------------------------

# remove trailing slash from BOOM_DIR
if BOOM_DIR.endswith("/"):
    BOOM_DIR = BOOM_DIR[:-1]

ext_modules = [
    Extension(
        '_boom',
        sources=boom_sources,
        include_dirs=[
            os.getcwd() + "/" + BOOM_DIR,
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++',
                       '-mmacosx-version-min=10.14',
                       '-Wno-sign-compare']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts
    elif sys.platform == 'linux':
        c_opts['unix'] = ['-Wno-sign-compare']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        print(f"compiler type is: {ct}")
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())  # noqa
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')

            # For deubgging purposes only.  Do not submit code with this option
            # present.
            # opts.append("-O0")
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())  # noqa
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


def FindPackagesAndBlab():
    # packages = ["R", "bsts", "spikeslab", "dynreg", "test_utils",
    # "boom/pybind11"]
    packages = find_namespace_packages(include=["BayesBoom.*"],
                                       exclude=["BayesBoom.*.*"])
    # packages = find_packages()
    if len(packages) == 0:
        packages = find_packages()
    print(f"***** HEY!! I found the following packages: {packages} *****")

    if len(packages) == 0:
        raise Exception("No packages found.")

    return packages


setup(
    name='BayesBoom',
    packages=FindPackagesAndBlab(),
    version=__version__,
    author='Steven L. Scott',
    author_email='steve.the.bayesian@gmail.com',
    url='https://github.com/steve-the-bayesian/BOOM',
    description='Tools for Bayesian modeling.',
    long_description="""Boom stands for 'Bayesian object oriented modeling'.
    It is also the sound your computer makes when it crashes.

    The main part of the Boom library is formulated in terms of abstractions
    for Model, Data, Params, and PosteriorSampler.  A Model is primarily an
    environment where parameters can be learned from data.  The primary
    learning method is Markov chain Monte Carlo, with custom samplers defined
    for specific models.

    The archetypal Boom program looks something like this:

    import BoomBayes as Boom

    some_data = 3 * np.random.randn(100) + 7
    model = Boom.GaussianModel()
    model.set_data(some_data)
    precision_prior = Boom.GammaModel(0.5, 1.5)
    mean_prior = Boom.GaussianModel(0, 10**2)
    poseterior_sampler = Boom.GaussianSemiconjugateSampler(
        model, mean_prior, precision_prior)
    model.set_method(poseterior_sampler)
    niter = 100
    mean_draws = np.zeros(niter)
    sd_draws = np.zeros(niter)
    for i in range(100):
        model.sample_posterior()
        mean_draws[i] = model.mu()
        sd_draws[i] = model.sigma()

    """,
    long_description_content_type="text/plain",
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.3'],
    setup_requires=['pybind11>=2.3'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
