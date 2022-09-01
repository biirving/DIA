from setuptools import setup, find_packages



setup(
    name = 'dia', 
    version = '0.0.1',
    author = 'Ben Irving',
    author_email = 'irving.b@northeastern.edu',
    description = 'A vision transformer library',
    url = 'https://github.com/Lysander-curiosum/DIA.git',
    package_dir = {"": "src"},
    packages = find_packages("src"), 
    packages = ['DIA'], 
)

