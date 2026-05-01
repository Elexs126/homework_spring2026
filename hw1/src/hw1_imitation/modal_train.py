from pathlib import Path
import tomllib

import modal

from hw1_imitation.train import TrainConfig, parse_train_config, run_training


APP_NAME = "hw1-imitation"
NETRC_PATH = Path("~/.netrc").expanduser()
PROJECT_DIR = "/root/project"
VOLUME_PATH = "/vol"
DEFAULT_GPU = "T4"
DEFAULT_CPU = 2.0
TORCH_LINUX_CU126 = (
    "torch @ "
    "https://download.pytorch.org/whl/cu126/"
    "torch-2.10.0%2Bcu126-cp311-cp311-manylinux_2_28_x86_64.whl"
    "#sha256=a9a9ba3b2baf23c044499ffbcbed88e04b6e38b94189c7dc42dd2cfcdd8c55c0"
)
volume = modal.Volume.from_name("hw1-imitation-volume", create_if_missing=True)


def project_dependencies() -> list[str]:
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as file:
        dependencies = tomllib.load(file)["project"]["dependencies"]
    return [
        TORCH_LINUX_CU126 if dependency.lower().startswith("torch") else dependency
        for dependency in dependencies
    ]


def load_gitignore_patterns() -> list[str]:
    """Translate .gitignore entries into Modal ignore globs."""

    if not modal.is_local():
        return []

    root = Path(__file__).resolve().parents[2]
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return []

    patterns: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            entry = entry.rstrip("/")
            patterns.append(f"**/{entry}/**")
        else:
            patterns.append(f"**/{entry}")
    return patterns


# Build a container image with the project's dependencies using uv pip.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(project_dependencies())
)
if NETRC_PATH.is_file():
    image = image.add_local_file(
        NETRC_PATH,
        remote_path="/root/.netrc",
        copy=True,
    )
image = image.add_local_dir(
    ".", remote_path=PROJECT_DIR, ignore=load_gitignore_patterns()
)


app = modal.App(APP_NAME)

env = {
    "PYTHONPATH": f"{PROJECT_DIR}/src",
    "WANDB_DIR": f"{VOLUME_PATH}/wandb",
}


@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=60 * 60 * 4,
    env=env,
    image=image,
    gpu=DEFAULT_GPU,
    cpu=DEFAULT_CPU,
)
def train_remote(*args: str) -> None:
    defaults = TrainConfig()
    defaults.data_dir = Path(VOLUME_PATH) / "data"
    config = parse_train_config(
        list(args),
        defaults=defaults,
        description="Train on Modal.",
    )
    run_training(config)
    volume.commit()
