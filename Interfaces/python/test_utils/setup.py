from setuptools import setup
import os

__version__ = '0.0.1'


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='test_utils',
    version=__version__,
    author='Steven L. Scott',
    author_email='steve.the.bayesian@gmail.com',
    url='https://github.com/steve-the-bayesian/BOOM',
    description='Utilities for writing unit tests for code involving MCMC',
    long_description=read("README"),
    packages=["test_utils"],
    zip_safe=True,
)
