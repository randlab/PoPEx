#!/usr/bin/env python

# Generic dependencies
from setuptools import setup, find_packages
import unittest


# Get unittest functions
def test_suite():
    test_loader = unittest.TestLoader()
    return test_loader.discover('tests', pattern='test_*.py')


# Long description from the README.md file
with open("README.md", "r") as file_handle:
    long_description = file_handle.read()


# Generic requirements
install_requires = ['numpy', 'scipy']

_setup_data = {
    'name':                 'popex',
    'version':              '1.0.0',
    'description':          'PoPEx sampling package',
    'long_description':      long_description,
    'long_description_content_type': 'text/markdown',

    'author':               'Christoph Jaeggli',
    'author_email':         'christoph.jaeggli@gmail.com',
    'url':                  'https://github.com/randlab/PoPEx',

    'license':              'MIT',

    'packages':             find_packages(),
    'install_requires':     install_requires,
}

if __name__ == '__main__':
    setup(**_setup_data)
