"""Compatibility shim for tools that still invoke setup.py directly."""

from setuptools import setup


if __name__ == "__main__":
    setup()
