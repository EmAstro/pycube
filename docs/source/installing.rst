=================
Installing pycube
=================

This section describes how to install pycube.

Installing Dependencies
=======================

There are a few packages that need to be installed before running the package.

We highly recommend that you use Anaconda for the majority of these installations.

Detailed installation instructions are presented below:

python and dependencies
-----------------------

pycube runs with `python <http://www.python.org/>`_ 3.9 and with the following dependencies:

* `python <http://www.python.org/>`_ -- version 3.9
* `astropy <https://www.astropy.org/>`_ -- version 4.2 or later
* `matplotlib <https://matplotlib.org/>`_ -- version 3.3 or later
* `numpy <http://www.numpy.org/>`_ -- version 1.20 or later
* `scipy <https://docs.scipy.org/doc/>`_ -- version 1.7 or later
* `astroquery <https://astroquery.readthedocs.io/en/latest/>`_ -- version 0.4 or later
* `photutils <https://photutils.readthedocs.io/en/stable/>`_ -- version 1.4 or later
* `extinction <https://extinction.readthedocs.io/en/latest/>`_ -- version 0.4 or later
* `sep <https://sep.readthedocs.io/en/v1.1.x/>`_ -- version 1.2 or later

If you are using Anaconda, you can check the presence of these packages with::

    conda list "^python$|astropy$|matplotlib$|numpy$"

If the packages have been installed, this command should print out all the packages and their version numbers.

If any of the packages are out of date, they can be updated with a command like::

    conda update astropy

The following packages are hosted by conda-forge. So the installation commands are::

    conda install -c conda-forge astroquery
    conda install -c conda-forge photutils
    conda install -c conda-forge extinction
    conda install -c conda-forge sep

git clone
---------

To install the package via GitHub run::

    git clone https://github.com/EmAstro/pycube.git

And, given that the packages is still work in progress and you may want to updated on-the-fly, we then recommend to install it with the `develop` option::

    cd pycube
    python setup.py develop

Testing the Installation
========================

In order to assess whether pycube has been properly installed, we suggest you run the following tests:

1. Run the default tests
------------------------

In the directory where pycube is installed run::

    pytest

You should see that all the current test passed.

2. Ensure that the scripts work
-------------------------------

Go to a directory outside of the pycube directory, then type pycube_collapse::

    cd
    pycube_collapse -h

