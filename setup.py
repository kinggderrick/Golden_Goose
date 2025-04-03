from setuptools import setup, find_packages

setup(
    name="goldengoose",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'MetaTrader5',
        'tensorflow>=2.10',
        'pandas',
        'numpy',
        'scikit-learn'
    ],
    entry_points={
        'console_scripts': [
            'goldengoose=bot:main',
        ],
    },
)