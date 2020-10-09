#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Based on https://github.com/navdeep-G/setup.py

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command
from setuptools.command.test import test as TestCommand

from grunnur import __version__


# Package meta-data.
NAME = 'grunnur'
DESCRIPTION = 'Uniform API for PyOpenCL and PyCUDA.'
URL = 'https://github.com/fjarri/grunnur'
EMAIL = 'bogdan@opanchuk.net'
AUTHOR = 'Bogdan Opanchuk'
REQUIRES_PYTHON = '>=3.7.0'
VERSION = __version__


REQUIRED = [
    'numpy>=1.6.0',
    'mako>=1.0.0',
    ]


EXTRAS = {
    'pyopencl': [
        'pyopencl>=2019.1.1',
        ],
    'pycuda': [
        'pycuda>=2019.1.1',
        ],
    'dev': [
        'pytest>=4',
        'pytest-cov',
        'sphinx>=2',
        'sphinx_autodoc_typehints',
        ]
    }


here = os.path.abspath(os.path.dirname(__file__))


try:
    with io.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        # Specifying the directory with tests explicitly
        # to prevent Travis CI from running tests from dependencies' eggs
        # (which are copied to the same directory).
        self.test_args = ['-x', 'test']
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/x-rst',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=[
        'grunnur',
        ],
    package_data={
        'grunnur': ['*.mako'],
        },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent',
        'Framework :: Pytest'
    ],
    cmdclass={
        'upload': UploadCommand,
        'test': PyTest,
    },
    entry_points={"pytest11": ["pytest_grunnur = grunnur.pytest_plugin"]},
)
