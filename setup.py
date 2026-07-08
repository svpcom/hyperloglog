#!/usr/bin/env python

from setuptools import setup

version = '0.1.8'

setup(
    name='hyperloglog',
    version=version,
    maintainer='Vasily Evseenko',
    maintainer_email='svpcom@gmail.com',
    author='Vasily Evseenko',
    author_email='svpcom@gmail.com',
    packages=['hyperloglog', 'hyperloglog.test'],
    description='HyperLogLog cardinality counter',
    url='https://github.com/svpcom/hyperloglog',
    install_requires=['msgpack', 'numpy'],
    python_requires='>=3.7',
    license='LGPL 2.1 or later',
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',
)
