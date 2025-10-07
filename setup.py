
from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read the contents of your requirements file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='crypto-predictor',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A hybrid Transformer-GAN model for cryptocurrency price prediction.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/your_username/crypto-predictor', # Replace with your repo URL
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'crypto-predictor=main:main',
        ],
    },
)
