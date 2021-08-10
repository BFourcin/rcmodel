from setuptools import setup, find_packages

setup(
    name='rcmodel',
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    url='https://github.com/mypackage.git',
    packages=find_packages(where='src'),
    package_dir={
        '': 'src'},
)
