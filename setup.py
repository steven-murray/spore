import setuptools
from numpy.distutils.core import setup, Extension
from setuptools import find_packages
import os
import sys
import re
import io

def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

if sys.argv[-1] == "publish":
    os.system("rm dist/*")
    os.system("python setup.py sdist")
    os.system("python setup.py bdist_wheel")
    os.system("twine upload dist/*")
    sys.exit()


resample = Extension('spore.fortran_routines.resample', ['spore/fortran_routines/resample.f90'],
                    extra_f90_compile_args=['-Wall', '-Wtabs', '-fopenmp'],
                    f2py_options=['only:', "lay_ps_map", ":"],
                    libraries=['gomp']
                    )

setup(
    name="spore",
    version=find_version("spore", "__init__.py"),
#    packages=['spore','spore.mock','spore.model','spore.measure','spore.visualise', ],
    install_requires=["numpy>=1.6.2",
                      "astropy>=1.0",
                      "scipy>=0.16",
                      "cached_property",
                      "powerbox==0.4.3"],
    author="Steven Murray",
    author_email="steven.murray@curtin.edu.au",
    description="Spectral Power Of the Reionisation Epoch",
    long_description=read('README.rst'),
    license="MIT",
    keywords="epoch of reionisation power spectrum",
    url="https://github.com/steven-murray/spore",
    ext_modules=[resample],
    packages=find_packages(),#['halomod', 'halomod.fort'] if os.getenv("WITH_FORTRAN", None) else ['halomod'],

    # could also include long_description, download_url, classifiers, etc.
)
