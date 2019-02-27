from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="cimcb_lite",
    version="0.0.5",
    description="This is a pre-release.",
    long_description=long_description,
    license="http://www.apache.org/licenses/LICENSE-2.0.html",
    url="https://github.com/KevinMMendez/cimcb_lite",
    packages=["cimcb_lite", "cimcb_lite.bootstrap", "cimcb_lite.cross_val", "cimcb_lite.model", "cimcb_lite.plot", "cimcb_lite.utils"],
    install_requires=[
        "numpy",
        "pandas",
        "bokeh",
        "scipy",
        "scikit-learn",
        "tqdm",
        "xlrd",
    ],
)
