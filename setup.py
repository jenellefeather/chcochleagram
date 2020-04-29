from setuptools import setup, find_packages

setup(
    name='chcochleagram',
    version='1.0',
    author='Jenelle Feather',
    author_email='jfeather@mit.edu',
    packages=find_packages(include=['chcochleagram', 'chcochleagram.*']),
    install_requires=[
        'numpy',
        'matplotlib',
        'jupyter',
        'torch',
        'scipy',
    ],
)

