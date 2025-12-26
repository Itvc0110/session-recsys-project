from setuptools import setup, find_packages

setup(
    name='session-recsys-project',
    version='0.1',
    packages=find_packages(),
    install_requires=['torch', 'numpy', 'pandas', 'pyyaml'],
)
