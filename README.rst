Python implementation of the Hyper LogLog and Sliding Hyper LogLog cardinality counter algorithms. 
--------------------------------------------------------------------------------------------------

Installation:
============

Use ``pip install hyperloglog`` to install from PyPI.

Usage:
=====

::

    import hyperloglog
    hll = hyperloglog.HyperLogLog(0.01)  # accept 1% counting error
    hll.add("hello")
    print len(hll)  # 1
    hll.add("hello")
    print len(hll)  # 1 as items aren't added more than once
    hll.add("hello again")
    print len(hll)  # 2

If we add a further 1000 random strings (giving a total of 1002 strings) we'll have a count roughly within 1% of the true value, in this case it counts 1007 (within +/- 10.2 of the true value)

::

    # add 1000 random 30 char strings to hll
    import random
    import string
    [hll.add("".join([string.ascii_letters[random.randint(0, len(string.ascii_letters)-1)] for n in range(30)])) for m in range(1000)]  
    print len(hll)  # 1007


Changes:
========
- Added Sliding window HLL version
- Added bias correction from HLL++

References:
===========

1. http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf
2. http://hal.archives-ouvertes.fr/docs/00/46/53/13/PDF/sliding_HyperLogLog.pdf
3. http://research.google.com/pubs/pub40671.html

Wiki: :ref:`Ask DeepWiki <https://deepwiki.com/badge.svg)](https://deepwiki.com/svpcom/hyperloglog>`
=======================
