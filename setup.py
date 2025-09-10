from setuptools import setup

setup(
    name="price_engine",
    version="2.0.0",
    author="Your Name",
    description="An advanced, end-to-end engine for price elasticity modeling and profit optimization.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=['price_engine'], 
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "statsmodels",
        "xgboost",
        "pygam"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
