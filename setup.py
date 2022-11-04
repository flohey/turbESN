from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

NAME = 'turbESN'
VERSION = None
AUTHOR = "flohey (Florian Heyder)"
EMAIL = "<florian.heyder@tu-ilmenau.de>"
DESCRIPTION = 'An echo state network implementation.'
LONG_DESCRIPTION = 'An echo state network implementation, used in my PhD research as part of the DeepTurb project of the Carl-Zeiss Stiftung.'

about = {}
if not VERSION:
    with open(os.path.join(here, NAME, "_version.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] =  VERSION


# Setting up
setup(
    name=NAME,
    version=about["__version__"],
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy', 'multiprocess', 'h5py', 'scipy'],
    include_package_data=True,
    package_data={NAME: ['*.json']},
    python_requires=">=3.7.0",
    keywords=['python', 'ESN', 'reservoir computing', 'echo state network', 'recurrent neural network'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
