from __future__ import annotations

from dataclasses import dataclass
import math

from .config import FactoryConfig
from .models import RobotSpec


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class FactoryCell:
    cell_id: str
    center: Vec3
    stage: str


@dataclass(frozen=True)
class ConveyorPath:
    conveyor_id: str
    points: tuple[Vec3, ...]


@dataclass(frozen=True)
class TransferZoneSpec:
    zone_id: str
    from_stage: str
    to_stage: str
    center: Vec3


@dataclass(frozen=True)
class FactoryLayout:
    cells: tuple[FactoryCell, ...]
    robots: tuple[RobotSpec, ...]
    main_conveyors: tuple[ConveyorPath, ConveyorPath]
    transfer_conveyor: ConveyorPath
    transfer_zones: tuple[TransferZoneSpec, ...]
    quality_control_area: Vec3
    packaging_areas: tuple[Vec3, Vec3]


_CELL_DEFS = (
    ("C1", (-6.0, 0.0, 0.0), "pick_place"),
    ("C2", (-3.0, 0.0, 0.0), "pick_place"),
    ("C3", (0.0, 0.0, 0.0), "assembly"),
    ("C4", (3.0, 0.0, 0.0), "joining"),
    ("C5", (6.0, 0.0, 0.0), "finishing"),
)

_ROBOT_OFFSETS = (
    (-0.85, -0.85, 0.0),
    (-0.85, 0.85, 0.0),
    (0.85, -0.85, 0.0),
    (0.85, 0.85, 0.0),
)


def _roles_for_cell(cell_id: str) -> tuple[str, str, str, str]:
    if cell_id in {"C1", "C2"}:
        return ("pick_place",) * 4
    if cell_id == "C3":
        return ("assembly",) * 4
    if cell_id == "C4":
        return ("joining",) * 4
    return ("quality_control", "quality_control", "packaging", "packaging")


def build_factory_layout(config: FactoryConfig) -> FactoryLayout:
    config.validate()
    if config.robot_count != 20 or config.cell_count != 5:
        raise ValueError("cinematic layout requires exactly 20 robots and 5 cells")

    cells = tuple(FactoryCell(cell_id, center, stage) for cell_id, center, stage in _CELL_DEFS)
    robots: list[RobotSpec] = []
    robot_index = 1

    for cell in cells:
        for role, offset in zip(_roles_for_cell(cell.cell_id), _ROBOT_OFFSETS):
            position = (
                cell.center[0] + offset[0],
                cell.center[1] + offset[1],
                cell.center[2] + offset[2],
            )
            yaw = math.atan2(cell.center[1] - position[1], cell.center[0] - position[0])
            robots.append(
                RobotSpec(
                    robot_id=f"R{robot_index:02d}",
                    cell_id=cell.cell_id,
                    role=role,
                    base_position=position,
                    base_yaw=yaw,
                )
            )
            robot_index += 1

    main_conveyors = (
        ConveyorPath("MAIN_IN", ((-8.0, -2.3, 0.6), (-4.5, -2.3, 0.6), (-2.0, -2.3, 0.6))),
        ConveyorPath("MAIN_OUT", ((4.5, 2.3, 0.6), (7.0, 2.3, 0.6), (8.5, 2.3, 0.6))),
    )
    transfer_conveyor = ConveyorPath(
        "TRANSFER",
        ((-5.0, 0.0, 0.62), (-2.5, 0.0, 0.62), (0.0, 0.0, 0.62), (3.0, 0.0, 0.62), (6.0, 0.0, 0.62)),
    )

    transfer_zones = (
        TransferZoneSpec("Z_PICK_ASSEMBLY", "pick_place", "assembly", (-1.5, 0.0, 0.75)),
        TransferZoneSpec("Z_ASSEMBLY_JOINING", "assembly", "joining", (1.5, 0.0, 0.75)),
        TransferZoneSpec("Z_JOINING_QC", "joining", "quality_control", (4.5, -0.35, 0.75)),
        TransferZoneSpec("Z_QC_PACKAGING", "quality_control", "packaging", (6.0, 0.55, 0.75)),
    )

    return FactoryLayout(
        cells=cells,
        robots=tuple(robots),
        main_conveyors=main_conveyors,
        transfer_conveyor=transfer_conveyor,
        transfer_zones=transfer_zones,
        quality_control_area=(5.5, -1.25, 0.6),
        packaging_areas=((6.6, 1.0, 0.6), (7.3, 1.0, 0.6)),
    )
