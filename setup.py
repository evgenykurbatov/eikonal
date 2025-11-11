from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

import eikonal
eikonal.__version__

setup(
    name='Eikonal',
    version=version,
    description='Eikonal-based wave optics',
    long_description=readme,
    author='Evgeny P. Kurbatov',
    author_email='evgeny.p.kurbatov@gmail.com',
    url='https://github.com/evgenykurbatov/eikonal',
    long_description_content_type='text/markdown',
    license=license,
    packages=find_packages(exclude=('examples', 'docs'))
)
