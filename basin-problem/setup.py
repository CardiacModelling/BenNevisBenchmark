from setuptools import setup, Extension
import pybind11

module_name = 'basin'
cpp_sources = ['basin.cpp']
include_dirs = [pybind11.get_include(),
                # '/usr/include/python3.10'
                ]

ext_modules = [
    Extension(
        language='c++',
        name=module_name,
        sources=cpp_sources,
        extra_compile_args=["-O3", "-fPIC", "-std=c++20", "-mcmodel=large"],
        include_dirs=include_dirs,
    )
]

setup(
    name=module_name,
    ext_modules=ext_modules,
    version='0.0.1',
    author='author',
)