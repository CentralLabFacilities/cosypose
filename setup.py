import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension
from os import path

setup(
    name='cosypose',
    version='1.0.0',
    description='CosyPose',
    packages=find_packages(),
    ext_modules=[
        CppExtension(
            name='cosypose_cext',
            sources=[
                'cosypose/csrc/cosypose_cext.cpp'
            ],
            extra_compile_args=['-O3'],
            verbose=True
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
