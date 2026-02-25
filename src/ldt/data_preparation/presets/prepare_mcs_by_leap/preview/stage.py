from __future__ import annotations

from beartype import beartype
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from ldt.utils.errors import InputValidationError

from ..stage_1_wave_paths import (
    available_waves as stage_1_available_waves,
)
from ..stage_1_wave_paths import (
    resolve_wave_dataset_config,
)
from ..stage_2_subjects import SubjectKeyConfig, subject_key_config
from ..stage_3_features import (
    LongitudinalFeatureMapping,
    NonLongitudinalFeatureSpec,
    resolve_longitudinal_mappings,
    resolve_non_longitudinal_specs,
)
from ..stage_4_composites import CompositeFeatureSpec, resolve_composite_features

_CONSOLE = Console()
_SUBJECT_KEY_GLOSSARY: dict[str, str] = {
    "MCSID": "MCS Research ID (anonymised family/household identifier).",
    "CNUM": "Cohort member number within MCSID (child index).",
    "PNUM": "Parent/respondent number within MCSID (parent index).",
    "CHID": "Synthetic child identifier built as `<MCSID>_<CNUM>`.",
    "wave": "Sweep label indicating the measurement occasion (W1..W7).",
}


@beartype
def render_prepare_mcs_by_leap_config_preview(*, waves: tuple[str, ...]) -> None:
    """Render a configuration-only preset preview without loading raw wave data.

    Args:
        waves (tuple[str, ...]): Waves.
    """

    ordered_waves = _normalise_preview_waves(waves=waves)
    keys = subject_key_config()

    _CONSOLE.rule("Prepare MCS by LEAP - Configuration Preview")
    _CONSOLE.print(
        _build_subject_construction_panel(
            keys=keys,
            waves=ordered_waves,
        )
    )
    _CONSOLE.print(_build_wave_role_summary_table(waves=ordered_waves))
    _CONSOLE.print(_build_longitudinal_config_table(waves=ordered_waves))
    _CONSOLE.print(_build_non_longitudinal_config_table(waves=ordered_waves))
    _CONSOLE.print(_build_composites_config_table(waves=ordered_waves))
    _CONSOLE.print(
        "[dim]Ambiguous source rule: scope filter first (if configured), "
        "then highest non-null coverage.[/dim]"
    )


@beartype
def _normalise_preview_waves(*, waves: tuple[str, ...]) -> tuple[str, ...]:
    """Validate and order preview wave labels."""

    if not waves:
        raise InputValidationError("Preview requires at least one wave.")

    supported = set(stage_1_available_waves())
    normalised: list[str] = []
    for wave in waves:
        token = wave.strip().upper()
        if token not in supported:
            supported_rendered = ", ".join(stage_1_available_waves())
            raise InputValidationError(
                f"Unsupported preview wave `{wave}`. Supported waves: {supported_rendered}."
            )
        if token not in normalised:
            normalised.append(token)

    return tuple(sorted(normalised, key=lambda item: int(item[1:])))


@beartype
def _build_subject_construction_panel(
    *,
    keys: SubjectKeyConfig,
    waves: tuple[str, ...],
) -> Panel:
    """Build a visual summary of child-level unique-subject construction logic."""

    tree = Tree(
        "[bold]Child-level unique subject per wave[/bold]",
        guide_style="bold magenta",
    )
    tree.add(f"[cyan]Child key[/cyan]: {' + '.join(keys.key_columns)}")
    tree.add(f"[cyan]Parent key[/cyan]: {' + '.join(keys.parent_key_columns)}")
    tree.add(f"[cyan]Link key[/cyan]: {' + '.join(keys.link_key_columns)}")
    tree.add(
        f"[cyan]Output fixed columns[/cyan]: {', '.join(keys.output_fixed_columns)}"
    )

    flow = tree.add("[bold]Construction flow[/bold]")
    flow.add("1) Build child anchor from union of child/link keys.")
    flow.add("2) Left-merge child-role datasets on child key.")
    flow.add("3) Left-merge family-role datasets on family key.")
    flow.add("4) Merge parent-role via link key and pivot PNUM to p1/p2/... slots.")
    flow.add(
        f"5) Add CHID as {keys.key_columns[0]}_{keys.key_columns[1]} and add wave label."
    )
    flow.add("6) Validate uniqueness at child key and guard against row explosion.")

    explanation = (
        "[bold]In a nutshell:[/bold]\n"
        "For each wave, we build one row per child using the child key "
        f"`{' + '.join(keys.key_columns)}`. "
        "We then merge child datasets on that child key, merge family datasets on "
        f"`{keys.key_columns[0]}`, and merge parent datasets through the link key "
        f"`{' + '.join(keys.link_key_columns)}` before pivoting parents into "
        "`p1`, `p2`, ... columns. "
        "This keeps a unique child-level subject while retaining family and parent context."
    )

    glossary_table = _build_subject_key_glossary_table(keys=keys)

    title = f"Stage 2 - Unique Subject Construction ({', '.join(waves)})"
    return Panel(
        Group(explanation, "", glossary_table, "", tree),
        title=title,
        border_style="magenta",
    )


@beartype
def _build_subject_key_glossary_table(*, keys: SubjectKeyConfig) -> Table:
    """Render a short glossary for configured subject-identity keys."""

    table = Table(
        title="Identifier Glossary",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Key", style="bold")
    table.add_column("Meaning")

    ordered_keys: list[str] = []
    for token in (
        *keys.output_fixed_columns,
        *keys.key_columns,
        *keys.parent_key_columns,
        *keys.link_key_columns,
    ):
        if token not in ordered_keys:
            ordered_keys.append(token)

    for token in ordered_keys:
        table.add_row(
            token, _SUBJECT_KEY_GLOSSARY.get(token, "No glossary entry configured.")
        )

    return table


@beartype
def _build_wave_role_summary_table(*, waves: tuple[str, ...]) -> Table:
    """Build stage-1/stage-2 role coverage summary from dataset configuration."""

    table = Table(
        title="Configured Wave Datasets by Role",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Wave")
    table.add_column("Datasets", justify="right")
    table.add_column("Family", justify="right")
    table.add_column("Child", justify="right")
    table.add_column("Parent", justify="right")
    table.add_column("Link", justify="right")
    table.add_column("Excluded", justify="right")

    for wave in waves:
        config = resolve_wave_dataset_config(wave)
        role_counts = {"family": 0, "child": 0, "parent": 0, "link": 0}
        for spec in config.datasets:
            role_counts[spec.role] += 1

        table.add_row(
            wave,
            f"{len(config.datasets):,}",
            f"{role_counts['family']:,}",
            f"{role_counts['child']:,}",
            f"{role_counts['parent']:,}",
            f"{role_counts['link']:,}",
            f"{len(config.excluded_datasets):,}",
        )
    return table


@beartype
def _build_longitudinal_config_table(*, waves: tuple[str, ...]) -> Table:
    """Build stage-3 longitudinal extraction matrix from configuration."""

    mappings_by_wave: dict[str, tuple[LongitudinalFeatureMapping, ...]] = {
        wave: resolve_longitudinal_mappings(wave) for wave in waves
    }
    canonical_index: dict[str, dict[str, LongitudinalFeatureMapping]] = {}
    for wave, mappings in mappings_by_wave.items():
        for mapping in mappings:
            canonical_index.setdefault(mapping.canonical, {})[wave] = mapping

    table = Table(
        title="Stage 3 - Longitudinal Features (Config Preview)",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_header=True,
        header_style="bold yellow",
    )
    table.add_column("Canonical", overflow="fold")
    for wave in waves:
        table.add_column(wave, overflow="fold")

    if not canonical_index:
        empty_row = ["(none configured)"] + ["-" for _ in waves]
        table.add_row(*empty_row)
        return table

    for canonical in sorted(canonical_index):
        row = [canonical]
        for wave in waves:
            mapping = canonical_index[canonical].get(wave)
            if mapping is None:
                row.append("-")
                continue
            scope = mapping.scope or "unspecified"
            row.append(f"{mapping.source} (scope: {scope})")
        table.add_row(*row)
    return table


@beartype
def _build_non_longitudinal_config_table(*, waves: tuple[str, ...]) -> Table:
    """Build stage-3 non-longitudinal extraction list from configuration."""

    table = Table(
        title="Stage 3 - Non-Longitudinal Features (Config Preview)",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_header=True,
        header_style="bold yellow",
    )
    table.add_column("Wave")
    table.add_column("Output")
    table.add_column("Source")
    table.add_column("Scope")

    rows = 0
    for wave in waves:
        specs: tuple[NonLongitudinalFeatureSpec, ...] = resolve_non_longitudinal_specs(
            wave
        )
        for spec in specs:
            rows += 1
            table.add_row(
                wave,
                spec.output,
                spec.source,
                spec.scope or "unspecified",
            )

    if rows == 0:
        table.add_row("(none configured)", "-", "-", "-")
    return table


@beartype
def _build_composites_config_table(*, waves: tuple[str, ...]) -> Table:
    """Build stage-4 composite feature configuration preview table."""

    table = Table(
        title="Stage 4 - Composite Features (Config Preview)",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Wave")
    table.add_column("Output")
    table.add_column("Operation")
    table.add_column("Inputs", overflow="fold")

    rows = 0
    for wave in waves:
        specs: tuple[CompositeFeatureSpec, ...] = resolve_composite_features(wave)
        for spec in specs:
            rows += 1
            table.add_row(
                wave,
                spec.output,
                spec.operation,
                ", ".join(spec.inputs),
            )

    if rows == 0:
        table.add_row("(none configured)", "-", "-", "-")
    return table
