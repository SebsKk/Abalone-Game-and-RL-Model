

try:
    import numpy
    print('Numpy version:', numpy.__version__)
except ImportError as e:
    print('Error importing Numpy:', e)
