import os
import re
import sys
import platform
import subprocess
import pdb
import glob

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")
        
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        
        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']


        env = os.environ.copy()
        cxx_append = ' -std=c++11 -stdlib=libc++ -std=c++1y' \
            if platform.system() == 'Darwin' else ''
        cxx_format = '{} -DVERSION_INFO=\\"{}\\"' + cxx_append
        env['CXXFLAGS'] = cxx_format.format(env.get('CXXFLAGS', ''), self.distribution.get_version())


        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        print(['cmake', ext.sourcedir] + cmake_args)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

        # directory hack for macOS
        if platform.system() == 'Darwin':
            for file in glob.glob(extdir + '*'):
                subprocess.check_call(['cp', file, './build'])


setup(
    name="diffqcqp",
    version='0.0.1',
    author="Quentin Le Lidec",
    author_email="quentin.le-lidec@inria.fr",
    url="https://github.com/quentinll/diffqcqp",
    description="Implementation of QP and QCQP solvers and their derivatives",
    long_description="",
    extras_require={},
    ext_modules=[CMakeExtension('diff_qcqp')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)