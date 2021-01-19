"""Setup script."""
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

if __name__ == "__main__":

    # Run setup
    setuptools.setup(
        name="bgnn",  # Replace with your own username
        version="0.0.1",
        author="Sergey Ivanov",
        author_email="sergei.ivanov@skolkovotech.ru",
        description="Boosted Graph Neural Networks",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/nd7141/bgnn",
        packages=setuptools.find_packages(),
        classifiers=[
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        python_requires='>=3.6',
    )