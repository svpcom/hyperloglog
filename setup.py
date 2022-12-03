#!/usr/bin/env python

from distutils.core import setup

version = '0.0.14'

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
    license='LGPL 2.1 or later',
    long_description=\
"""
Python implementation of the Hyper LogLog and Sliding Hyper LogLog cardinality counter
algorithms. Added bias correction from HLL++.

http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf
http://hal.archives-ouvertes.fr/docs/00/46/53/13/PDF/sliding_HyperLogLog.pdf
http://research.google.com/pubs/pub40671.html
""",
)
