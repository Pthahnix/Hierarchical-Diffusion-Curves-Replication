from setuptools import setup, find_packages

setup(
    name="hierarchical_diffusion_curves",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "kornia>=0.7.0",
        "scipy>=1.10.0",
        "numpy>=1.24.0",
        "pillow>=9.5.0",
    ],
)
