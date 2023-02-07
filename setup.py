import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lucidsonicdreams", 
    version="0.01",
    author="BigZaddy",
    author_email="@gmail.com",
    description="Syncs GAN-generated visuals to music",
    long_description=long_description,
    long_description_content_type="text/markdown",
   # url="https://github.com/takerdao2/lucid-sonic-dreams",
   # download_url="https://github.com/takertao2/lucid-sonic-dreams/lucid-sonic-dreams.tar.gz",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['click',
                      'moviepy==0.2.3.5',                    
                      'requests',
                      'tqdm',
                      'ninja',
                      'imageio==2.4.1',
                      'imageio-ffmpeg==0.4.3',
                      'pyspng==0.1.0']
)
