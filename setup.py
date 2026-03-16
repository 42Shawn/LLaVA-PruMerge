from setuptools import setup, find_packages

setup(
    name='llava_bitnet_prumerge',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.31.0",
        "accelerate>=0.21.0",
        "numpy",
        "scipy",
        "Pillow",
        "peft",
        "bitsandbytes",
        "deepspeed",
        "matplotlib",
        "xformers",
    ],
    entry_points={
        'console_scripts': [],
    },
)



