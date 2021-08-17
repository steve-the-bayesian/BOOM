from setuptools import setup
import os


MAJOR = 0
MINOR = 0
PATCH = 1


__version__ = f'{MAJOR}.{MINOR}.{PATCH}'


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='BayesBoom.dynreg',
    version=__version__,
    author='Steven L. Scott',
    author_email='steve.the.bayesian@gmail.com',
    url='https://github.com/steve-the-bayesian/BOOM',
    description='Sparse dynamic regression models.',
    long_description=read("README"),
    packages=["BayesBoom.dynreg"],
    zip_safe=True,
)
