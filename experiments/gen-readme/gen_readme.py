"""
This script generates a README.md file for your repo.

It works by combining all of the text in your repo into a single prompt sent to the ChatGPT API.
"""

from pathlib import Path

import click
import openai
import pathspec
import tiktoken


@click.command()
@click.argument("repo_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--model",
    default="gpt-4",
    type=click.Choice(["gpt-4", "gpt-3.5-turbo"]),
    help="Which model to use for generation.",
)
def cli(repo_dir, model):
    """
    Generate README.md content for the repo at REPO_DIR.
    """
    repo_dir = Path(repo_dir)

    # Read the .gitignore file if present
    gitignore_path = repo_dir / ".gitignore"
    spec = None
    if gitignore_path.exists():
        spec = pathspec.PathSpec.from_lines(
            "gitwildmatch", gitignore_path.read_text().splitlines()
        )

    # Traverse through the directory tree and collect text, respecting the .gitignore if present
    text = ""
    # TODO: Does adding filenames improve the quality of the generated README?
    for path in repo_dir.rglob("*"):
        if not path.is_file():
            continue
        if spec and spec.match_file(path):
            continue
        if path.parts[0] == ".git":
            continue
        if path.name == ".gitignore":
            continue

        click.echo(path, err=True)
        try:
            text += path.read_text() + "\n"
        except UnicodeDecodeError:
            pass

    # Create a prompt
    prompt = (
        "Create a README.md file for a repository. The repository contains the following files:\n"
        f"{text}"
    )

    # Estimate cost
    # Note that this excludes a few tokens e.g. to distinguish messages
    encoder = tiktoken.encoding_for_model("gpt-4")
    num_tokens = len(encoder.encode(prompt))
    cost_per_token = 0.004 / 1000  # Conservative estimate
    cost = num_tokens * cost_per_token

    click.echo(f"Number of tokens: {num_tokens}", err=True)
    click.echo(f"Estimated cost: ${cost:.2f}", err=True)

    # Get confirmation from the user
    if not click.confirm("Continue?", err=True):
        return

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    # Generate README.md
    result = response.choices[0].text.strip()  # type: ignore
    click.echo(result)


if __name__ == "__main__":
    cli()
