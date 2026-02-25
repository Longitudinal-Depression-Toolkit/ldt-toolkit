from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from beartype import beartype


@beartype
@dataclass(frozen=True)
class OneVsRestScenario:
    """One-vs-rest binary classification scenario derived from one target.

    Attributes:
        scenario_key (str): Identifier for scenario key.
        scenario_label (str): Scenario label.
        positive_class_label (str): Positive class label.
        y_binary (pd.Series): Y binary.

    """

    scenario_key: str
    scenario_label: str
    positive_class_label: str
    y_binary: pd.Series


@beartype
class TargetScenarioPlanner:
    """Build deterministic target scenarios for classification workflows."""

    @staticmethod
    @beartype
    def build_one_vs_rest_scenarios(
        *,
        y: pd.Series,
        scenario_prefix: str = "traj",
    ) -> tuple[OneVsRestScenario, ...]:
        """Create one-vs-rest binary targets for each unique class label.

        Args:
            y (pd.Series): Target labels.
            scenario_prefix (str): Prefix used when naming generated scenarios.

        Returns:
            tuple[OneVsRestScenario, ...]: Tuple of resolved values.
        """

        unique_labels = sorted(pd.unique(y).tolist(), key=str)
        scenario_keys_seen: set[str] = set()
        scenarios: list[OneVsRestScenario] = []
        for index, class_label in enumerate(unique_labels, start=1):
            label_text = str(class_label)
            base_key = (
                f"{scenario_prefix}_"
                f"{TargetScenarioPlanner._slugify(label=label_text)}_vs_rest"
            )
            scenario_key = base_key
            while scenario_key in scenario_keys_seen:
                scenario_key = f"{base_key}_{index}"
            scenario_keys_seen.add(scenario_key)

            y_binary = (y == class_label).astype("int64")
            scenarios.append(
                OneVsRestScenario(
                    scenario_key=scenario_key,
                    scenario_label=f"{label_text} vs rest",
                    positive_class_label=label_text,
                    y_binary=pd.Series(y_binary, index=y.index, name=y.name),
                )
            )
        return tuple(scenarios)

    @staticmethod
    @beartype
    def _slugify(*, label: str) -> str:
        """Convert one label to a filesystem-safe slug."""

        normalised = re.sub(r"[^a-z0-9]+", "_", label.lower())
        normalised = normalised.strip("_")
        return normalised or "class"
