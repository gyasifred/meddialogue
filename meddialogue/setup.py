"""
MedDialogue Package Setup
========================

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="meddialogue",
    version="1.0.0",
    author="Frederick Gyasi",
    author_email="gyasi@musc.edu",
    description="Healthcare Conversational Fine-Tuning Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/musc-bic/meddialogue",
    project_urls={
        "Bug Tracker": "https://github.com/musc-bic/meddialogue/issues",
        "Documentation": "https://meddialogue.readthedocs.io",
        "Source Code": "https://github.com/musc-bmic/meddialogue",
        "Institution": "https://medicine.musc.edu/departments/biomedical-informatics",
    },
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch==2.8.0",
        "transformers==4.55.4",
        "datasets==3.6.0",
        "unsloth==2025.9.7",
        "peft==0.17.1",
        "trl==0.22.2",
        "pandas==2.3.0",
        "numpy==1.26.4",
        "tqdm==4.67.1",
        "scikit-learn==1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "jupyter>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "meddialogue-train=meddialogue.cli:train_command",
            "meddialogue-infer=meddialogue.cli:infer_command",
        ],
    },
    include_package_data=True,
    package_data={
        "meddialogue": [
            "templates/*.json",
            "configs/*.yaml",
        ],
    },
    zip_safe=False,
    keywords=[
        "healthcare",
        "medical-ai",
        "fine-tuning",
        "conversational-ai",
        "clinical-nlp",
        "llm",
        "transformers",
        "lora",
        "patient-safety",
        "data-multiplication",
        "pii-detection",
        "bias-monitoring",
        "medical-terminology",
    ],
)