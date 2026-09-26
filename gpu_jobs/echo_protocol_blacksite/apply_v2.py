from pathlib import Path

ROOT = Path("unity_action_game")
TOOLS = ROOT / "tools"
BUILDER = ROOT / "Assets" / "Game" / "Editor" / "AutoSceneBuilder.cs"
README = ROOT / "README.md"

fetcher = r'''#!/usr/bin/env python3
from __future__ import annotations
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".asset-cache"
EXTERNAL = ROOT / "Assets" / "External"
SCIFI = EXTERNAL / "Quaternius_SciFiEssentials"
CHARS = EXTERNAL / "Quaternius_AnimatedCharacters"

PACKS = [
    (
        "scifi_models",
        "https://opengameart.org/sites/default/files/sci-fi_essentials_kit_models.zip",
        SCIFI,
    ),
    (
        "scifi_textures",
        "https://opengameart.org/sites/default/files/sci-fi_essentials_kit_textures.zip",
        SCIFI,
    ),
    (
        "characters",
        "https://opengameart.org/sites/default/files/ultimate_animated_character_pack_by_quaternius.zip",
        CHARS,
    ),
]
ALLOWED = {".fbx", ".obj", ".mtl", ".png", ".jpg", ".jpeg", ".tga"}

def download(url: str, out: Path) -> None:
    if out.is_file() and out.stat().st_size > 0:
        return
    print("Downloading", url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 EchoProtocolCI"})
    with urllib.request.urlopen(req, timeout=180) as response, out.open("wb") as f:
        shutil.copyfileobj(response, f)

def extract_filtered(archive: Path, dest: Path) -> int:
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            suffix = Path(info.filename).suffix.lower()
            if suffix not in ALLOWED:
                continue
            safe = Path(*[p for p in Path(info.filename).parts if p not in {"..", "/", "\\"}])
            target = dest / safe
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count

CACHE.mkdir(parents=True, exist_ok=True)
if SCIFI.exists():
    shutil.rmtree(SCIFI)
if CHARS.exists():
    shutil.rmtree(CHARS)
SCIFI.mkdir(parents=True, exist_ok=True)
CHARS.mkdir(parents=True, exist_ok=True)

copied = {}
for name, url, dest in PACKS:
    archive = CACHE / (name + ".zip")
    download(url, archive)
    copied[name] = extract_filtered(archive, dest)

def count_models(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in {".fbx", ".obj"})

def count_textures(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tga"})

scifi_models = count_models(SCIFI)
character_models = count_models(CHARS)
textures = count_textures(EXTERNAL)
total_models = scifi_models + character_models

source = f"""Echo Protocol external 3D sources
Creator: Quaternius
License: CC0 1.0 Universal

Pack 1: Sci-Fi Essentials Kit (Standard)
Source page: https://opengameart.org/content/sci-fi-essentials-kit
Models archive: {PACKS[0][1]}
Textures archive: {PACKS[1][1]}
Sci-fi model files detected: {scifi_models}

Pack 2: Animated Characters Pack
Source page: https://opengameart.org/content/animated-characters-pack
Characters archive: {PACKS[2][1]}
Animated character model files detected: {character_models}

Total model files detected: {total_models}
Total textures detected: {textures}
"""
(EXTERNAL / "ASSET_SOURCE.txt").write_text(source, encoding="utf-8")

if scifi_models < 5:
    raise SystemExit(f"Expected at least 5 sci-fi model files; found {scifi_models}")
if character_models < 10:
    raise SystemExit(f"Expected at least 10 external humanoid character model files; found {character_models}")

print({
    "scifi_models": scifi_models,
    "humanoid_character_models": character_models,
    "total_models": total_models,
    "textures": textures,
    "copied": copied,
})
'''

validator = r'''#!/usr/bin/env python3
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "Assets" / "Game" / "Scripts"
EXTERNAL = ROOT / "Assets" / "External"
SCIFI = EXTERNAL / "Quaternius_SciFiEssentials"
CHARS = EXTERNAL / "Quaternius_AnimatedCharacters"

required = [
    "PlayerController.cs", "ThirdPersonCamera.cs", "WeaponSystem.cs", "Health.cs",
    "EnemyAI.cs", "MissionDirector.cs", "StoryDirector.cs", "ObjectiveZone.cs",
    "ArchiveTerminal.cs", "GameHUD.cs", "GameSession.cs",
]
errors = []
for name in required:
    if not (SCRIPTS / name).is_file():
        errors.append(f"missing C# file: {name}")

mission = (SCRIPTS / "MissionDirector.cs").read_text(encoding="utf-8") if (SCRIPTS / "MissionDirector.cs").exists() else ""
for phase in ("Infiltrate", "RetrieveArchive", "SurviveAmbush", "Extraction", "Complete"):
    if phase not in mission:
        errors.append(f"mission phase missing: {phase}")

player = (SCRIPTS / "PlayerController.cs").read_text(encoding="utf-8") if (SCRIPTS / "PlayerController.cs").exists() else ""
for token in ("Horizontal", "Vertical", "LeftShift", "Jump", "KeyCode.E"):
    if token not in player:
        errors.append(f"player control token missing: {token}")

weapon = (SCRIPTS / "WeaponSystem.cs").read_text(encoding="utf-8") if (SCRIPTS / "WeaponSystem.cs").exists() else ""
for token in ("GetMouseButton(0)", "KeyCode.R", "Physics.Raycast", "TakeDamage"):
    if token not in weapon:
        errors.append(f"weapon behavior token missing: {token}")

for cs in sorted(SCRIPTS.glob("*.cs")):
    source = re.sub(r"//.*", "", cs.read_text(encoding="utf-8"))
    if source.count("{") != source.count("}"):
        errors.append(f"brace mismatch: {cs.name}")
    if "namespace EchoProtocol" not in source:
        errors.append(f"namespace missing: {cs.name}")

builder_path = ROOT / "Assets" / "Game" / "Editor" / "AutoSceneBuilder.cs"
builder = builder_path.read_text(encoding="utf-8") if builder_path.exists() else ""
for token in ("Quaternius_AnimatedCharacters", "Player_External_Humanoid_Visual", "Quaternius_SciFiEssentials"):
    if token not in builder:
        errors.append(f"scene builder external-model constraint missing: {token}")

def model_files(path: Path):
    return [p for p in path.rglob("*") if p.suffix.lower() in {".fbx", ".obj"}] if path.exists() else []

models = model_files(EXTERNAL)
scifi = model_files(SCIFI)
chars = model_files(CHARS)
textures = [p for p in EXTERNAL.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tga"}] if EXTERNAL.exists() else []

if len(scifi) < 5:
    errors.append(f"too few sci-fi external models: {len(scifi)}")
if len(chars) < 10:
    errors.append(f"too few external humanoid character models: {len(chars)}")

record = EXTERNAL / "ASSET_SOURCE.txt"
if not record.exists():
    errors.append("asset source record missing")
else:
    rec = record.read_text(encoding="utf-8")
    for token in ("Sci-Fi Essentials Kit", "Animated Characters Pack", "CC0"):
        if token not in rec:
            errors.append(f"asset source record missing: {token}")

handmade = [p for p in (ROOT / "Assets" / "Game").rglob("*") if p.suffix.lower() in {".fbx", ".obj", ".blend", ".gltf", ".glb"}]
if handmade:
    errors.append("authored 3D model found under Assets/Game: " + ", ".join(str(p) for p in handmade[:5]))

report = {
    "game": "Echo Protocol: Blacksite",
    "validation": "failed" if errors else "passed",
    "csharp_files": len(list(SCRIPTS.glob("*.cs"))),
    "external_models": len(models),
    "sci_fi_models": len(scifi),
    "humanoid_character_models": len(chars),
    "external_textures": len(textures),
    "external_sources": [
        "Quaternius Sci-Fi Essentials Kit / CC0",
        "Quaternius Animated Characters Pack / CC0",
    ],
    "errors": errors,
}
(ROOT / "build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
if errors:
    sys.exit(1)
'''

TOOLS.mkdir(parents=True, exist_ok=True)
(TOOLS / "fetch_external_assets.py").write_text(fetcher, encoding="utf-8")
(TOOLS / "validate_project.py").write_text(validator, encoding="utf-8")

s = BUILDER.read_text(encoding="utf-8")
old_player = '''            string model = PickModel(paths, new[] { "character", "player", "soldier", "human", "agent", "male", "female" }, new[] { "enemy", "robot", "turret", "weapon", "gun" });
            AddExternalVisual(player.transform, model, 1.75f, "Player_External_Visual");'''
new_player = '''            List<string> characterModels = paths
                .Where(p => p.IndexOf("Quaternius_AnimatedCharacters", StringComparison.OrdinalIgnoreCase) >= 0)
                .ToList();
            if (characterModels.Count == 0)
                throw new InvalidOperationException("Animated Characters Pack is missing; ECHO-7 must use an externally downloaded humanoid model.");

            string model = PickModel(characterModels, new[] { "male", "female", "human", "ninja", "cowboy", "character" }, new[] { "zombie", "goblin", "elf" });
            AddExternalVisual(player.transform, model, 1.75f, "Player_External_Humanoid_Visual");'''
old_enemy = '''            string model = PickModel(paths, new[] { "enemy", "robot", "hazmat", "character", "soldier" }, new[] { "weapon", "gun" });'''
new_enemy = '''            List<string> sciFiModels = paths
                .Where(p => p.IndexOf("Quaternius_SciFiEssentials", StringComparison.OrdinalIgnoreCase) >= 0)
                .ToList();
            if (sciFiModels.Count == 0)
                throw new InvalidOperationException("Sci-Fi Essentials Pack is missing; enemy visuals cannot be assembled.");
            string model = PickModel(sciFiModels, new[] { "enemy", "robot", "hazmat" }, new[] { "weapon", "gun", "prop" });'''

if old_player not in s:
    raise SystemExit("player builder block not found")
if old_enemy not in s:
    raise SystemExit("enemy builder block not found")
BUILDER.write_text(s.replace(old_player, new_player).replace(old_enemy, new_enemy), encoding="utf-8")

readme = README.read_text(encoding="utf-8")
readme = readme.replace(
    "Visible models and textures come from **Quaternius — Sci-Fi Essentials Kit (Standard)**, mirrored by OpenGameArt under **CC0 1.0**. The pack includes sci-fi props, guns and animated robot enemies. No custom FBX/OBJ/Blend models are authored by this project.",
    "Visible models and textures come from two external **Quaternius** packs mirrored by OpenGameArt under **CC0 1.0**: **Sci-Fi Essentials Kit (Standard)** for the blacksite/robots/weapons and **Animated Characters Pack** for ECHO-7. No custom FBX/OBJ/Blend models are authored by this project."
)
README.write_text(readme, encoding="utf-8")
print("Echo Protocol v2 patch applied.")
