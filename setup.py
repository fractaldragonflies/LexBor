import pathlib
import codecs
from setuptools import setup, find_packages

# The directory containing this file
LOCAL_PATH = pathlib.Path(__file__).parent

# The text of the README file
README_CONTENTS = (LOCAL_PATH / "README.md").read_text()

# Load requirements, so they are listed in a single place
REQUIREMENTS_PATH = LOCAL_PATH / "requirements.txt"
with open(REQUIREMENTS_PATH.as_posix()) as fp:
    install_requires = [dep.strip() for dep in fp.readlines()]

setup(
    author_email="fractaldragonflies@gmail.com",
    author="John E. Miller",
    description="A Python library for lexical borrowing detection.",
    extras_require={
        "examples": ["pyclts", "cldfbench", "pylexibank"],
        "tests": ["pytest", "pytest-cov"],
    },
    include_package_data=True,
    install_requires=install_requires,
    keywords="borrowing, language contact, sequence comparison, language model, classifier",
    license="Apache License 2.0",
    long_description_content_type="text/markdown",
    long_description=codecs.open("README.md", "r", "utf-8").read(),
    name="lexbor",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    url="https://github.com/fractaldragonflies/lexbor/",
    version="1.0",  # remember to sync with __init__.py
    zip_safe=False,
)
