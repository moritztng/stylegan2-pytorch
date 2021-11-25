import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stylegan2",
    version="0.1",
    author="Moritz Thuening",
    author_email="moritz.thuening@gmail.com",
    description="tiny stylegan2 implementation in pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moritztng/stylegan2-pytorch",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.10.0',
        'torchvision>=0.11.1',
        'gdown>=4.2.0',
        'jinja2>=2.11.3',
        'click>=7.1.2'
    ],
    entry_points={
        "console_scripts": [
            "stylegan2 = stylegan2.__main__:stylegan2"
        ]
    }
)
