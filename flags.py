"""
Command-line flags generated from a dataclass config: every simple field becomes a flag with the same name
(`npc_drive_weight` -> `--npc-drive-weight`), so a run changes only the settings it passes and everything else keeps
the dataclass's defaults.

    parser = argparse.ArgumentParser()
    add_dataclass_flags(parser, GridworldConfig)
    config = dataclass_from_flags(parser.parse_args(), GridworldConfig)
"""

import argparse
from dataclasses import Field, fields

FLAG_TYPES = (bool, int, float, str)  # fields of these types get a flag; tuples, dicts etc. don't


def parse_bool(text: str) -> bool:
    """Reads a boolean flag's value: true/false, 1/0 or yes/no."""
    if text.lower() in ("true", "1", "yes"):
        return True
    if text.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")


def _flag_fields(config_cls: type) -> list[Field]:
    """The fields of `config_cls` that get a flag."""
    return [config_field for config_field in fields(config_cls) if config_field.type in FLAG_TYPES]


def add_dataclass_flags(parser: argparse.ArgumentParser, config_cls: type, title: str | None = None) -> None:
    """
    Adds one flag per simple field of `config_cls`, named after it, in its own section of `--help`. Flags default to
    None, meaning "keep the dataclass's default", which `--help` shows.
    """
    group = parser.add_argument_group(title or f"{config_cls.__name__} settings")
    defaults = config_cls()
    for config_field in _flag_fields(config_cls):
        group.add_argument(
            f"--{config_field.name.replace('_', '-')}",
            type=parse_bool if config_field.type is bool else config_field.type,
            default=None,
            metavar=config_field.type.__name__.upper(),
            help=f"default: {getattr(defaults, config_field.name)}",
        )


def dataclass_from_flags(cli: argparse.Namespace, config_cls: type):
    """An instance of `config_cls` with every flag that was passed applied on top of its defaults."""
    passed = {
        config_field.name: getattr(cli, config_field.name)
        for config_field in _flag_fields(config_cls)
        if getattr(cli, config_field.name) is not None
    }
    return config_cls(**passed)
