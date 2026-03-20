from __future__ import annotations

import os
import os.path as osp
import shutil

VALIDATED_WRITEUP_NAME = "template.validated.tex"


def validated_writeup_path(latex_folder: str) -> str:
    return osp.join(latex_folder, VALIDATED_WRITEUP_NAME)


def has_validated_writeup(latex_folder: str) -> bool:
    return osp.exists(validated_writeup_path(latex_folder))


def restore_validated_writeup(latex_folder: str, writeup_file: str) -> bool:
    snapshot = validated_writeup_path(latex_folder)
    if not osp.exists(snapshot):
        return False
    shutil.copyfile(snapshot, writeup_file)
    return True


def save_validated_writeup(latex_folder: str, latex_text: str) -> None:
    snapshot = validated_writeup_path(latex_folder)
    os.makedirs(latex_folder, exist_ok=True)
    with open(snapshot, "w", encoding="utf-8") as f:
        f.write(latex_text)
