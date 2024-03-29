import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# taken from https://github.com/CEA-COSMIC/ModOpt/blob/master/setup.py
with open('requirements.txt') as open_file:
    install_requires = open_file.read()

__version__ = "develop"
setuptools.setup(
    name="tf-noisy-image-dataset",
    version=__version__,
    author="Zaccharie Ramzi",
    author_email="zaccharie.ramzi@gmail.com",
    description="Datasets for noisy images in the tf.data API.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zaccharieramzi/tf-noisy-image-dataset",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    python_requires='>=3.6',
)
