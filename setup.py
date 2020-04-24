import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fourier_accountant",
    version="0.11",
    author="Antti Koskela",
    author_email="anttik123@gmail.com",
    description="Fourier Accountant for Differential Privacy ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    tests_require=[],
    test_suite='tests',
    url="https://github.com/DPBayes/PLD-Accountant",
    packages=setuptools.find_packages(include=['fourier_accountant', 'fourier_accountant.*']),
    install_requires=[
        'numpy'
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)
