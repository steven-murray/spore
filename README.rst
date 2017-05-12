spore
=====

**Spectral Power Of the Reionization Epoch**

This is a code to model statistical foregrounds in the Epoch of Reionization. The underlying approach is equivalent to
CHIPS (the Cosmological HI Power Spectrum estimator, see Trott et al, http://iopscience.iop.org/article/10.3847/0004-637X/818/2/139/pdf).
However, the philosophy behind this code is somewhat different.

Code Philosophy
~~~~~~~~~~~~~~~
The idea behind ``spore`` is to provide a set of tools which can calculate relevant statistical quantities in a plug-n-play
manner. This means that most parameters are open to the user to modify. Furthermore, it means that the included models
themselves -- such as beam models, source counts, spatial distribution etc. -- are in principle modifiable by the user,
such that the overall framework will automatically include them.

Features
~~~~~~~~

* Consistent unit definitions across quantities
* Consistent conversion of units between radio astronomy and cosmology.
* Calculate 2D power-spectrum covariance analytically for arbitrary input models and scales.
* Calculate 2D power-spectrum covariance via sampling from statistical distributions, for arbitrary input models and scales.
* Some support for working with numerical simulations (from 21cmFAST)
* Some visualisation tools (in dev.)
* Intuitive package structure, separating the ideas of "measuring", "modelling", "mocking" and "visualising" the 2D power spectrum.


Installation
~~~~~~~~~~~~
**Non-python-users**

If you don't use python much, I suggest installing the Anaconda python distribution for your OS, and ideally creating
an env for using spore::


    $ conda create --name spore_env numpy scipy astropy
    $ activate spore_env


Once you've done this, continue to the next step...

**Python users**

If you already/now have a working python environment, you *should* just be able to do the following::


    $ pip install git+git://github.com/steven-murray/spore.git


All the dependencies should be automatically installed.


Usage
~~~~~
I'll update this later, but for now, have a look at the notebook ``devel/paper_plots.ipynb``.


