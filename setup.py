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
tests_require = ['numpy']
install_requires = [p_req for p_req in tests_require]


_setup_data = {
    'name':                 'spo_mds',
    'version':              '0.1.0',
    'description':          'PoPEx sampling package',
    'long_description':     long_description,

    'author':               'Christoph Jaeggli',
    'author_email':         'christoph.jaeggli@gmail.com',
    'url':                  '',

    'packages':             find_packages(),
    'test_suite':           'setup.test_suite',
    'tests_require':        tests_require,
    'install_requires':     install_requires,
}

if __name__ == '__main__':
    setup(**_setup_data)
