import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="timedisagg", # Replace with your own username
    version="0.0.1",
    author="John Selvam",
    author_email="jstephenj14@gmail.com",
    description="Temporal Disaggregation of Low-Frequency Time Series Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jstephenj14/timedisagg",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)