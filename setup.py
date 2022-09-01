from setuptools import setup, find_packages



setup(
    name = 'DecepticonsInAlexandria', 
    version = '0.0.3',
    author = 'Ben Irving',
    author_email = 'irving.b@northeastern.edu',
    description = 'A vision transformer library',
    url = 'https://github.com/Lysander-curiosum/DIA',
    package_dir = {"": "src"},
    packages = find_packages("src"), 
)

