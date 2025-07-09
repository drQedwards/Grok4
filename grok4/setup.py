from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="grok4",
    version="0.1.0",
    author="Grok4 Team",
    author_email="team@grok4.ai",
    description="Advanced AI Framework for Language Understanding and Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/grok4/grok4",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "training": [
            "deepspeed>=0.9.0",
            "wandb>=0.15.0",
            "flash-attn>=2.0.0",
        ],
        "inference": [
            "triton>=2.0.0",
            "bitsandbytes>=0.41.0",
            "xformers>=0.0.20",
        ],
    },
    entry_points={
        "console_scripts": [
            "grok4=grok4.cli:main",
            "grok4-train=grok4.training.train:main",
            "grok4-serve=grok4.api.server:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)