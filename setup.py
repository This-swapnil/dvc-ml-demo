from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.1",
    author="swapnil sonawane",
    description="A small package for dvc ml pipline demo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/This-swapnil/dvc-ml-demo",
    author_email="sswapnil0098@gmail.com",
    packages=["src"],
    license="GNU",
    python_requires=">=3.9",
    install_requires=["pandas", "scikit-learn", "dvc"],
)