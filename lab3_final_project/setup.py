from setuptools import setup, find_packages

setup(
    name="soil_retention",
    version="0.1",
    description="Analysis of soil water retention curves",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "scipy", "matplotlib", "seaborn", "scikit-learn"],
)
