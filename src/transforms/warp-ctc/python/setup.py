from distutils.core import setup
setup(name='ctc',
      version='0.1',
      packages=['ctc'],
      package_dir={'ctc': '.'},
      package_data={'ctc' : ['../build/libwarpctc.*']}
      )
