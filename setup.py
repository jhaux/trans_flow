from setuptools import setup, find_packages

setup(
    name='tflow',
    version='0.0.1',
    # url='https://github.com/mypackage.git',
    author='Johannes Haux',
    author_email='jo.mobile.2011@gmail.com',
    description='Toy example for invertible NNs',
    packages=find_packages(),    
    install_requires=['numpy', 'torch', 'edflow'],
)
