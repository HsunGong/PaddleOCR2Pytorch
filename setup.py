from setuptools import setup, find_packages

setup(
    name='pytorchocr',
    version='0.1.0',
    description='OCR inference package based on PyTorch',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(include=['pytorchocr', 'pytorchocr.*']),
    install_requires=[
        'numpy>=1.24',
        'pillow',
        'pyclipper',
        'shapely',
        'opencv-python>=4.1.0',
        'torch>=1.13',
    ],
    python_requires='>=3.7',
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
