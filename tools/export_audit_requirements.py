"""Export exact project pins for dependency auditing without resolving wheels."""

import argparse
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from packaging.requirements import Requirement


def exact_pins(pyproject_path: Path) -> list[str]:
    with pyproject_path.open("rb") as file:
        project = tomllib.load(file)["project"]

    dependencies = list(project.get("dependencies", []))
    for group in project.get("optional-dependencies", {}).values():
        dependencies.extend(group)

    pins = set()
    for dependency in dependencies:
        requirement = Requirement(dependency)
        specifiers = list(requirement.specifier)
        if requirement.url or len(specifiers) != 1:
            continue
        specifier = specifiers[0]
        if specifier.operator == "==" and "*" not in specifier.version:
            pins.add(f"{requirement.name}=={specifier.version}")
    return sorted(pins, key=str.casefold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    args = parser.parse_args()
    args.output.write_text("\n".join(exact_pins(args.pyproject)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
