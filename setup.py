from setuptools import setup, find_packages, Command

# Load text for description and license
with open('README.md') as f:
    readme = f.read()

# Go!
setup(
    # Module name (lowercase)
    name='IPA',
    version='0.0.0dev0',

    # Description
    description='Implementation of the Information Profile Algorithm (IPA).',
    long_description=readme,

    # License name
    license='GPL',

    # Maintainer information
    maintainer='David Augustin',
    maintainer_email='david.augustin@cs.ox.ac.uk',
    url='https://github.com/DavAug/IPA.git',

    # Packages to include
    packages=find_packages(include=('IPA','IPA.*')),

    # List of dependencies
    install_requires=[
        'cma>=2',
        'numpy>=1.8',
        'scipy>=0.14',
        'matplotlib>=1.5',
        'myokit>=1.29',
        'tabulate',
        'pandas',
        'pints @ git+git://github.com/pints-team/pints.git#egg=pints'
    ],
    dependency_links=[
     "git+git://github.com/pints-team/pints.git#egg=pints-0.2.2",
    ]
)