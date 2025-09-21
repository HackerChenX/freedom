"""
VnPy EFinance数据源模块安装配置
"""

from setuptools import setup, find_packages

# 读取README文件
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vnpy_efinance",
    version="1.0.0",
    author="VnPy Freedom",
    author_email="support@vnpy.com", 
    description="EFinance数据源 for VnPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vnpy/vnpy_efinance",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "vnpy",
        "efinance>=0.5.5",
        "pandas>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
        ]
    },
    package_data={
        "vnpy_efinance": ["*.py"],
    },
    keywords="vnpy efinance trading quant finance data stock",
)
