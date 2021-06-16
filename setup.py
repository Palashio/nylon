import setuptools
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="nylon-ai",  # Replace with your own username
    version="0.0.5",
    author="Palash Shah",
    author_email="hello@paraglide.ai",
    description="An interactive grammar of machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Palashio/nylon",
    install_requires=requirements,
    packages=find_packages(exclude=('tests',)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)