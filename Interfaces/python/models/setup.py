from setuptools import setup, find_namespace_packages
import os

MAJOR = 0
MINOR = 0
PATCH = 1

__version__ = f'{MAJOR}.{MINOR}.{PATCH}'


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='BayesBoom.models',
    version=__version__,
    author='Steven L. Scott',
    author_email='steve.the.bayesian@gmail.com',
    url='https://github.com/steve-the-bayesian/BOOM',
    license='MIT',
    description='Python wrappers for BayesBoom C++ model objects.',
    packages=find_namespace_packages(include=["BayesBoom.*"]),
    install_requires=['BayesBoom'],
    zip_safe=True,
)
