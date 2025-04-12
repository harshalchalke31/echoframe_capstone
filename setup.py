from setuptools import setup, find_packages

setup(
    name="echoframe",
    version="0.1.1",
    packages=find_packages(),  # Automatically finds 'src' and 'pretraining' as packages
    package_dir={
        "src": "src",  # Map 'src' as a package
        "pretraining": "pretraining",  # Map 'pretraining' as another package
        "src_3d": "src_3d"
    },
)