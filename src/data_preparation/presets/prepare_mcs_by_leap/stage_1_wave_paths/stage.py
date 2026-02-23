from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import yaml
from beartype import beartype

from src.utils.errors import InputValidationError

DatasetRole = Literal["family", "child", "parent", "link"]

_STAGE_DIR = Path(__file__).resolve().parent
_DATASETS_CONFIG = _STAGE_DIR / "datasets.yaml"


@beartype
@dataclass(frozen=True)
class DatasetSpec:
    """Specification for one raw dataset required for a wave.

    Attributes:
        file_name (str): Name for file.
        alias (str): Alias.
        role (DatasetRole): Role.
        cnum_source (str | None): Cnum source.
        pnum_source (str | None): Pnum source.
        filter_to_valid_child_cnum (bool): Whether to filter to valid child cnum.
        required_identifiers (tuple[str, ...]): Required identifiers.
    """

    file_name: str
    alias: str
    role: DatasetRole
    cnum_source: str | None = None
    pnum_source: str | None = None
    filter_to_valid_child_cnum: bool = False
    required_identifiers: tuple[str, ...] = ("MCSID",)


@beartype
@dataclass(frozen=True)
class WaveDatasetConfig:
    """Configuration payload for one wave of raw MCS datasets.

    Attributes:
        wave (str): Wave.
        datasets (tuple[DatasetSpec, ...]): Datasets.
        excluded_datasets (tuple[str, ...]): Excluded datasets.
    """

    wave: str
    datasets: tuple[DatasetSpec, ...]
    excluded_datasets: tuple[str, ...] = ()


@beartype
@cache
def available_waves() -> tuple[str, ...]:
    """Return supported wave labels in numeric order.

    Returns:
        tuple[str, ...]: Tuple of resolved values.
    """

    return tuple(sorted(_wave_configs().keys(), key=lambda value: int(value[1:])))


@beartype
def resolve_wave_dataset_config(wave: str) -> WaveDatasetConfig:
    """Return parsed dataset configuration for one wave.

    This helper is used by higher-level workflows and keeps input/output handling consistent.

    Args:
        wave (str): Wave identifier.

    Returns:
        WaveDatasetConfig: Result object for this operation.
    """

    normalised = wave.strip().upper()
    configs = _wave_configs()
    if normalised not in configs:
        supported = ", ".join(available_waves())
        raise InputValidationError(
            f"Unsupported wave `{wave}`. Supported waves: {supported}."
        )
    return configs[normalised]


@beartype
def expected_files_for_wave(wave: str) -> tuple[str, ...]:
    """Return expected raw file names for one wave.

    Args:
        wave (str): Wave identifier.

    Returns:
        tuple[str, ...]: Tuple of resolved values.
    """

    return tuple(spec.file_name for spec in resolve_wave_dataset_config(wave).datasets)


@beartype
def validate_wave_raw_path(*, wave: str, raw_dir: Path) -> None:
    """Validate that a raw input directory matches the configured wave files.

    Args:
        wave (str): Wave identifier.
        raw_dir (Path): Filesystem location for raw dir.
    """

    if not raw_dir.exists() or not raw_dir.is_dir():
        raise InputValidationError(
            f"Raw directory does not exist for {wave}: {raw_dir.resolve()}"
        )

    config = resolve_wave_dataset_config(wave)
    missing = [
        spec.file_name
        for spec in config.datasets
        if not (raw_dir / spec.file_name).exists()
    ]
    if not missing:
        return

    expected = "\n".join(
        f"- {spec.file_name}"
        for spec in sorted(config.datasets, key=lambda item: item.file_name)
    )
    found = "\n".join(
        f"- {path.name}"
        for path in sorted(raw_dir.glob("*.dta"), key=lambda item: item.name)
    )
    missing_rendered = "\n".join(f"- {name}" for name in sorted(missing))
    raise InputValidationError(
        f"At `{raw_dir.resolve()}`, required files are missing for {wave}.\n"
        f"Missing files:\n{missing_rendered}\n"
        f"Expected files for this wave:\n{expected}\n"
        f"Found `.dta` files:\n{found if found else '- (none)'}"
    )


@beartype
@cache
def _wave_configs() -> dict[str, WaveDatasetConfig]:
    """Load wave raw-dataset configuration from stage YAML."""

    raw = _load_yaml_config(_DATASETS_CONFIG)
    waves = raw.get("waves")
    if not isinstance(waves, dict) or not waves:
        raise InputValidationError(
            "Stage 1 datasets config must define non-empty `waves` mapping."
        )

    parsed: dict[str, WaveDatasetConfig] = {}
    for wave_label, payload in waves.items():
        if not isinstance(wave_label, str) or not wave_label.strip():
            raise InputValidationError("Wave labels must be non-empty strings.")
        wave = wave_label.strip().upper()
        if not isinstance(payload, dict):
            raise InputValidationError(f"Wave `{wave}` config must be a mapping.")

        raw_datasets = payload.get("datasets")
        if not isinstance(raw_datasets, list) or not raw_datasets:
            raise InputValidationError(
                f"Wave `{wave}` must define a non-empty `datasets` list."
            )

        dataset_specs = tuple(
            _parse_dataset_spec(item=item, wave=wave) for item in raw_datasets
        )
        excluded = _parse_string_list(
            value=payload.get("excluded_datasets", []),
            context=f"excluded_datasets for {wave}",
        )
        parsed[wave] = WaveDatasetConfig(
            wave=wave,
            datasets=dataset_specs,
            excluded_datasets=excluded,
        )
    return parsed


@beartype
def _parse_dataset_spec(*, item: Any, wave: str) -> DatasetSpec:
    """Parse one dataset entry from stage-1 configuration."""

    if not isinstance(item, dict):
        raise InputValidationError(
            f"Invalid dataset entry for {wave}. Expected a mapping."
        )

    file_name = _parse_required_string(
        item, key="file_name", context=f"dataset entry in `{wave}`"
    )
    role_raw = _parse_required_string(
        item,
        key="role",
        context=f"dataset `{file_name}` in `{wave}`",
    )
    role = _parse_role(role_raw=role_raw, context=f"dataset `{file_name}` in `{wave}`")

    alias_raw = item.get("alias")
    alias = (
        alias_raw.strip()
        if isinstance(alias_raw, str) and alias_raw.strip()
        else Path(file_name).stem
    )
    cnum_source = _parse_optional_string(item.get("cnum_source"))
    pnum_source = _parse_optional_string(item.get("pnum_source"))
    filter_to_valid_child_cnum = bool(item.get("filter_to_valid_child_cnum", False))

    required_raw = item.get("required_identifiers")
    if required_raw is None:
        required_identifiers = _build_required_identifiers(
            cnum_source=cnum_source,
            pnum_source=pnum_source,
        )
    else:
        required_identifiers = _parse_string_list(
            value=required_raw,
            context=f"required_identifiers for dataset `{file_name}`",
        )

    return DatasetSpec(
        file_name=file_name,
        alias=alias,
        role=role,
        cnum_source=cnum_source,
        pnum_source=pnum_source,
        filter_to_valid_child_cnum=filter_to_valid_child_cnum,
        required_identifiers=required_identifiers,
    )


@beartype
def _build_required_identifiers(
    *,
    cnum_source: str | None,
    pnum_source: str | None,
) -> tuple[str, ...]:
    """Build default required identifiers from configured key sources."""

    required = ["MCSID"]
    if cnum_source is not None:
        required.append(cnum_source)
    if pnum_source is not None:
        required.append(pnum_source)
    return tuple(required)


@beartype
def _parse_required_string(data: dict[str, Any], *, key: str, context: str) -> str:
    """Parse one required non-empty string value."""

    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Missing `{key}` in {context}.")
    return value.strip()


@beartype
def _parse_optional_string(value: Any) -> str | None:
    """Parse one optional non-empty string value."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise InputValidationError("Optional config value must be a string.")
    stripped = value.strip()
    return stripped or None


@beartype
def _parse_string_list(*, value: Any, context: str) -> tuple[str, ...]:
    """Parse one list of non-empty strings."""

    if not isinstance(value, list):
        raise InputValidationError(f"Expected list for `{context}`.")
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise InputValidationError(f"Invalid string entry in `{context}`.")
        parsed.append(item.strip())
    return tuple(parsed)


@beartype
def _parse_role(*, role_raw: str, context: str) -> DatasetRole:
    """Parse one dataset role string."""

    role = role_raw.strip().lower()
    if role not in {"family", "child", "parent", "link"}:
        raise InputValidationError(
            f"Invalid role `{role_raw}` in {context}. "
            "Expected one of: family, child, parent, link."
        )
    return role  # type: ignore[return-value]


@beartype
@cache
def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load one YAML config file from disk and validate mapping root."""

    if not config_path.exists() or not config_path.is_file():
        raise InputValidationError(f"Missing config file: {config_path.resolve()}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise InputValidationError(
            f"Config file `{config_path.name}` must contain a YAML mapping."
        )
    return loaded
