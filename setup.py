from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tfl-hpl",
    version="1.0.0",
    author="Burra Deepak Yadav",
    author_email="deepakyadavdeepu94@gmail.com",
    description="Trustworthy Federated Learning with Heterogeneous Privacy Levels for Critical Infrastructure",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deepakdeepu-12/TFL-HPL",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.13.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "cryptography>=41.0.0",
        "pysyft>=0.6.0",
    ],
)