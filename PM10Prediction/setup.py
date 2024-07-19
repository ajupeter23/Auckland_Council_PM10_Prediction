from setuptools import setup, find_packages

setup(
    name='PM10Prediction',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'tensorflow',
        'matplotlib',
        'scikit-learn',
    ],
    include_package_data=True,
    description='A package for PM10 prediction using various machine learning models',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/PM10Prediction',  # Update this with your repository URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify the minimum Python version
)
