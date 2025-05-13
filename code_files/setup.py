from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="neuro240",
    version="0.1.0",
    description="LLM reasoning enhancement using Classical RL",
    author="Dinesh Vasireddy",
    author_email="dineshvasireddy@college.harvard.edu",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 