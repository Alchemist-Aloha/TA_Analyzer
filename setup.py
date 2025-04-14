from setuptools import setup, find_packages

setup(
    name="TA_Analyzer",
    version="1.0.0",
    description="A package for analyzing transient absorption spectroscopy data.",
    author="Likun Cai",
    url="https://github.com/Alchemist-Aloha/TA_Analyzer",  # Replace with your repository URL
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "xarray",
        "lmfit",
        "scipy",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)