"""
Protein thermostability prediction using ESM2 model fine-tuned with LoRA.
CLI tool for predicting melting temperatures (Tm) from protein sequences.
"""

from setuptools import setup, find_packages

short_description = __doc__.split("\n")
try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except FileNotFoundError:
    long_description = "\n".join(short_description[2:])


setup(
    name="tmprot",
    author="Karen Pailozian, Pavel Kohout, Loschmidt Laboratories",
    author_email="karen.pailozian@fnusa.cz",
    version="1.0.0",
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/loschmidt/TmProt",
    license="LGPLv3",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0",
        "torch>=2.0",
        "transformers>=4.30",
        "peft>=0.4",
        "biopython>=1.81",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "tmprot=tmprot.cli:predict",
        ],
    },
)
