from pathlib import Path

from setuptools import setup


def get_long_description():
    with (Path(__file__).parent / "README.md").open() as fp:
        return fp.read()


def get_version():
    g = {}
    with (Path(__file__).parent / "codal" / "version.py").open() as fp:
        exec(fp.read(), g)
    return g["__version__"]


setup(
    name="codal",
    description="Understand large code repos with LLMs",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Wasim Lorgat",
    url="https://github.com/seem/codal",
    project_urls={
        "Issues": "https://github.com/seem/codal/issues",
        "CI": "https://github.com/seem/codal/actions",
        "Changelog": "https://github.com/seem/codal/releases",
    },
    license="Apache License, Version 2.0",
    version=get_version(),
    packages=["codal"],
    entry_points="""
        [console_scripts]
        codal=codal.cli:cli
    """,
    install_requires=[
        "alembic",
        "click",
        "GitPython",
        "langchain",
        "numpy",
        "openai",
        "SQLAlchemy",
        "tiktoken",
        "tqdm",
    ],
    extras_require={"test": ["pytest", "cogapp", "sqlalchemy[mypy]", "types-tqdm"]},
    python_requires=">=3.8",
)
