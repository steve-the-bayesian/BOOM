from setuptools import setup, find_namespace_packages
import os

MAJOR = 0
MINOR = 0
PATCH = 1

__version__ = f'{MAJOR}.{MINOR}.{PATCH}'


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='BayesBoom.factormodels',
    version=__version__,
    author='Steven L. Scott',
    author_email='steve.the.bayesian@gmail.com',
    url='https://github.com/steve-the-bayesian/BOOM',
    description='Bayesian methods for models defined by latent factors.',
    long_description=read("README"),
    packages=find_namespace_packages(include=["BayesBoom.*"]),
    zip_safe=True,
)
