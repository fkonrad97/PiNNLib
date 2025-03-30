from setuptools import setup, find_packages

setup(
    name="pinnLib",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21",
        "scipy>=1.7",
        "matplotlib",
    ],
    author="fkonrad97",
    description="Physics-Informed Neural Network library for solving PDEs like Black-Scholes",
)