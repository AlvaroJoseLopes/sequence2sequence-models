import setuptools
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='seq2seq_models',
    version='0.0.1',
    author='Alvaro José Lopes',
    author_email='',
    description='Seq2Seq models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/AlvaroJoseLopes/sequence2sequence-models/tree/main/seq2seq',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=['torch', 'torchtext', 'torchdata']
)