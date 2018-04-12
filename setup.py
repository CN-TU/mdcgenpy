from setuptools import setup, find_packages
from codecs import open
from os import path
from pip.req import parse_requirements
import pip.download


install_reqs = parse_requirements("requirements.txt", session=pip.download.PipSession())
install_requires = [str(ir.req) for ir in install_reqs]


here = path.abspath(path.dirname(__file__))


# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mdcgenpy',
    version='1.0.0',

    description='MDCGen (Multidimensional Dataset for Clustering Generator)',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/CN-TU/mdcgenpy',

    # Author details
    author='Daniel C. Ferreira',
    author_email='daniel.ferreira@nt.tuwien.ac.at',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # What does your project relate to?
    keywords='data generation clustering',

    packages=['mdcgenpy'],

    install_requires=install_requires
)
