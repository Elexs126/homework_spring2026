from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple


Vec3 = Tuple[float, float, float]


def _parse_vec3(text: str) -> Vec3:
    parts = text.split()
    if len(parts) != 3:
        raise ValueError(f"Expected 3 floats, got: {text!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _format_vec3(v: Vec3) -> str:
    # Keep it compact but stable.
    return f"{v[0]:.10g} {v[1]:.10g} {v[2]:.10g}"


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _parse_fromto(text: str) -> tuple[Vec3, Vec3]:
    parts = text.split()
    if len(parts) != 6:
        raise ValueError(f"Expected 6 floats for fromto, got: {text!r}")
    a = (float(parts[0]), float(parts[1]), float(parts[2]))
    b = (float(parts[3]), float(parts[4]), float(parts[5]))
    return a, b


def _format_fromto(a: Vec3, b: Vec3) -> str:
    return f"{_format_vec3(a)} {_format_vec3(b)}"


def _adjust_subtree_not_bodies(elem: ET.Element, body_global: Vec3) -> None:
    if elem.tag == "body":
        return

    pos = elem.get("pos")
    if pos is not None:
        elem.set("pos", _format_vec3(_sub(_parse_vec3(pos), body_global)))

    fromto = elem.get("fromto")
    if fromto is not None:
        a, b = _parse_fromto(fromto)
        elem.set("fromto", _format_fromto(_sub(a, body_global), _sub(b, body_global)))

    for child in list(elem):
        _adjust_subtree_not_bodies(child, body_global)


def _get_body_global_pos(body: ET.Element) -> Vec3:
    pos = body.get("pos")
    if pos is None:
        # In these Gym assets the relevant bodies always have explicit pos.
        return (0.0, 0.0, 0.0)
    return _parse_vec3(pos)


def _convert_body_tree(body: ET.Element, parent_global: Vec3) -> None:
    body_global = _get_body_global_pos(body)

    # Convert this body's pos from global to local.
    body.set("pos", _format_vec3(_sub(body_global, parent_global)))

    # Convert all non-body descendants that belong to this body's frame.
    for child in list(body):
        if child.tag != "body":
            _adjust_subtree_not_bodies(child, body_global)

    # Recurse into nested bodies using the ORIGINAL global positions.
    for child_body in [c for c in list(body) if c.tag == "body"]:
        _convert_body_tree(child_body, body_global)


@dataclass(frozen=True)
class PatchResult:
    file: Path
    patched: bool
    backup: Path | None


def patch_gym_mujoco_asset(xml_path: Path, *, backup_suffix: str = ".orig") -> PatchResult:
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)

    raw = xml_path.read_text(encoding="utf-8")
    if 'coordinate="global"' not in raw:
        return PatchResult(file=xml_path, patched=False, backup=None)

    backup_path = xml_path.with_suffix(xml_path.suffix + backup_suffix)
    if not backup_path.exists():
        shutil.copy2(xml_path, backup_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Remove coordinate="global" from the top-level compiler tag.
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("coordinate") == "global":
        compiler.attrib.pop("coordinate", None)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"No <worldbody> found in {xml_path}")

    # Convert each top-level body.
    for top_body in [c for c in list(worldbody) if c.tag == "body"]:
        _convert_body_tree(top_body, (0.0, 0.0, 0.0))

    tree.write(xml_path, encoding="utf-8", xml_declaration=False)
    return PatchResult(file=xml_path, patched=True, backup=backup_path)


def _default_asset_paths() -> list[Path]:
    import gym  # type: ignore

    gym_root = Path(gym.__file__).resolve().parent
    assets_dir = gym_root / "envs" / "mujoco" / "assets"
    return [assets_dir / "hopper.xml", assets_dir / "walker2d.xml"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Patch Gym 0.25 MuJoCo assets that use compiler coordinate=global so they work with mujoco>=3.x. "
            "Creates .orig backups next to each file."
        )
    )
    parser.add_argument(
        "--files",
        nargs="*",
        type=Path,
        default=None,
        help="Optional explicit xml files to patch (default: hopper.xml + walker2d.xml inside installed gym)",
    )

    args = parser.parse_args(argv)

    files = args.files if args.files is not None else _default_asset_paths()

    any_patched = False
    for xml_path in files:
        try:
            result = patch_gym_mujoco_asset(xml_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {xml_path}: {exc}", file=sys.stderr)
            return 1

        if result.patched:
            any_patched = True
            print(f"[PATCHED] {result.file} (backup: {result.backup})")
        else:
            print(f"[SKIP] {result.file} (no coordinate=global)")

    if not any_patched:
        print("No files needed patching.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
