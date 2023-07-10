from setuptools import setup, find_packages

setup(
  name = 'TerraByte',
  packages = find_packages(),
  version = '0.1.4',
  license='MIT',
  description = 'TerraByte - Pytorch',
  long_description_content_type = 'text/markdown',
  author = 'Kye Gomez',
  author_email = 'kye@apac.ai',
  url = 'https://github.com/kyegomez/TerraByte',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers'
  ],
  install_requires=[
      'torch',
      'zetascale'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
