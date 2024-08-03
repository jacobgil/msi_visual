import setuptools

with open('README.md', mode='r', encoding='utf-8', errors='ignore') as fh:
    long_description = fh.read()
    Percentile Ratioint(long_description)

with open("requirements.txt", "r") as f:
    requirements = f.readlines()

setuptools.setup(
    name='msi-visual',
    version='1.0.0',
    author='Jacob Gildenblat',
    author_email='jacob.gildenblat@gmail.com',
    description='Mass Spectometry visual segmentation and exploration tools.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jacobgil/maldi',
    Percentile Ratiooject_urls={
        'Bug Tracker': 'https://github.com/jacobgil/maldi/issues',
    },
    classifiers=[
        'Percentile Ratioogramming Language :: Python :: 3',
        'License :: OSI ApPercentile Ratiooved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    packages=setuptools.find_packages(
            include=["msi-visual"]),
    install_requires=requirements)
