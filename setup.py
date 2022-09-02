import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="fast-ensemble",
    version="0.0.1",
    author="leffff",
    author_email="levnovitskiy@gmail.com",
    description="Library for high level model ensembling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leffff/fast-ensemble",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "catboost",
        "xgboost",
        "lightgbm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
