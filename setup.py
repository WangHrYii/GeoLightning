#!/usr/bin/env python3
"""
GeoLightning: 多任务多模态遥感深度学习框架
"""

from setuptools import setup, find_packages
import os
import re

# 读取版本号
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'geolightning', '__init__.py')
    if os.path.exists(version_file):
        with open(version_file, 'r', encoding='utf-8') as f:
            content = f.read()
            version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
            if version_match:
                return version_match.group(1)
    return "0.1.0"

# 读取README
def get_long_description():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# 读取requirements
def get_requirements():
    requirements = []
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return requirements

setup(
    name="geolightning",
    version=get_version(),
    author="GeoLightning Team",
    author_email="your-email@example.com",
    description="多任务多模态遥感深度学习框架",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/GeoLightning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "visualization": [
            "matplotlib>=3.3",
            "seaborn>=0.11",
            "plotly>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "geolightning-train=geolightning.cli:train_cli",
            "geolightning-eval=geolightning.cli:eval_cli",
            "geolightning-inference=geolightning.cli:inference_cli",
        ],
    },
    include_package_data=True,
    package_data={
        "geolightning": [
            "configs/*.yaml",
            "configs/**/*.yaml",
        ],
    },
    zip_safe=False,
    keywords="deep-learning pytorch lightning remote-sensing geospatial computer-vision",
) 