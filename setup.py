from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="cimcb_lite",
    version="0.1.6",
    description="This is a pre-release.",
    long_description=long_description,
    license="http://www.apache.org/licenses/LICENSE-2.0.html",
    url="https://github.com/KevinMMendez/cimcb_lite",
    packages=[
        "cimcb_lite",
        "cimcb_lite.bootstrap",
        "cimcb_lite.cross_val",
        "cimcb_lite.model",
        "cimcb_lite.plot",
        "cimcb_lite.utils"],
    python_requires='>=3.5',
    install_requires=[
        "bokeh>=1.0.0",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "tqdm",
        "xlrd"],
    author='Kevin Mendez, David Broadhurst',
    author_email='k.mendez@ecu.edu.au, d.broadhurst@ecu.edu.au',
)
