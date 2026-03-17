import json
import pickle
from pathlib import Path
from typing import Any

from .agent_manager import AgentManager, StageTransition
from .journal import Journal


def load_stage_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    with open(checkpoint_file, "rb") as handle:
        checkpoint = pickle.load(handle)
    required_keys = {
        "journals",
        "stage_history",
        "task_desc",
        "current_stage",
    }
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing required keys: {sorted(missing_keys)}"
        )
    return checkpoint


def build_resumed_manager(
    checkpoint_path: str | Path,
    cfg: Any,
    workspace_dir: Path,
) -> tuple[AgentManager, dict[str, Any]]:
    checkpoint = load_stage_checkpoint(checkpoint_path)
    task_desc = checkpoint["task_desc"]
    manager = AgentManager(
        task_desc=json.dumps(task_desc),
        cfg=cfg,
        workspace_dir=workspace_dir,
    )

    completed_stage = checkpoint["current_stage"]
    journals = dict(checkpoint["journals"])
    if completed_stage.name not in journals:
        raise ValueError(
            f"Checkpoint journal does not contain stage: {completed_stage.name}"
        )

    next_stage = manager._create_next_main_stage(
        completed_stage,
        journals[completed_stage.name],
    )
    if next_stage is None:
        raise ValueError(
            f"Checkpoint stage {completed_stage.name} is already the final stage."
        )

    stage_history = list(checkpoint["stage_history"])
    stage_history.append(
        StageTransition(
            from_stage=completed_stage.name,
            to_stage=next_stage.name,
            reason=f"Resumed from checkpoint {Path(checkpoint_path).name}",
            config_adjustments={"resume_checkpoint": str(Path(checkpoint_path))},
        )
    )

    journals[next_stage.name] = Journal()
    manager.task_desc = task_desc
    manager.journals = journals
    manager.stage_history = stage_history
    manager.stages = [completed_stage, next_stage]
    manager.current_stage = next_stage
    manager.current_stage_number = next_stage.stage_number
    manager.completed_stages = list(checkpoint["journals"].keys())
    return manager, task_desc
