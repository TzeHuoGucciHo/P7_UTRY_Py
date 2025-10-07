from pathlib import Path

root = Path("body_measure")
# Folders
(root / "data").mkdir(parents=True, exist_ok=True)
(root / "src" / "body_measure").mkdir(parents=True, exist_ok=True)
(root / "tests").mkdir(parents=True, exist_ok=True)

# Empty files
for p in [
    root / "pyproject.toml",
    root / "README.md",
    root / ".gitignore",
    root / "data" / ".gitkeep",
    root / "src" / "body_measure" / "__init__.py",
    root / "src" / "body_measure" / "cli.py",
    root / "src" / "body_measure" / "measure.py",
    root / "src" / "body_measure" / "segmenter.py",
    root / "src" / "body_measure" / "pose.py",
    root / "src" / "body_measure" / "calibrate.py",
    root / "src" / "body_measure" / "geometry.py",
    root / "tests" / "test_geometry.py",
    root / "tests" / "test_smoke.py",
]:
    p.touch()

print(f"Created empty project at: {root.resolve()}")
