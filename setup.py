from setuptools import setup, find_packages

setup(name='dlocr',

      version='0.1-SNAPSHOT',

      url='https://github.com/GlassyWing/text-detection-ocr',

      license='Apache 2.0',

      author='Manlier',

      author_email='dengjiaxim@gmail.com',

      description='ocr base on deep learning',

      keywords='OCR,detect,Chinese word recognition',

      packages=find_packages(exclude=['tests', 'tools']),

      package_data={'dlocr': ['*.*', 'weights/*', 'config/*', 'dictionary/*']},

      long_description=open('README.md', encoding='utf-8').read(),

      zip_safe=False,

      install_requires=['pandas', 'keras', 'xmltodict', 'opencv-python', 'matplotlib', 'pillow'],

      )
