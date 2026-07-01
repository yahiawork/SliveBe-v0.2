import base64
import json
import math
import os
import random
import re
import sys
import time
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import pygame
import urllib.error
import urllib.request

SCREEN_TITLE = "SliveBe M14 - 2.5D"
VIEW_WIDTH = 1280
VIEW_HEIGHT = 720
PROFILE_PATH = Path("uses/player_profile.json")
MENU_BG_PATH = Path("uses/img.png")
MENU_TITLE_PATH = Path("uses/ti.png")
PLAY_BTN_PATH = Path("uses/play_btn.png")
CUSTOMIZATION_BTN_PATH = Path("uses/customization_btn.png")
EXIT_BTN_PATH = Path("uses/exit_btn.png")
PLAYER_IMAGE_PATH = Path("uses/player.png")

WORLD_SIZE = 64
TILE_SIZE = 64

PLAYER_RADIUS = 8
PLAYER_MAX_SPEED = 255.0
PLAYER_ACCEL = 1750.0
PLAYER_FRICTION = 2450.0
PLAYER_JUMP_SPEED = 0.0
GRAVITY = 1800.0
GAME_TIME_LIMIT = 240.0
SPEED_ITEM_DURATION = 10.0
SPEED_ITEM_MULTIPLIER = 2.0

BULLET_SPEED = 710.0
BULLET_LIFETIME = 1.8

CAMERA_LAG = 10.0
CAMERA_LEAD = 0.16

HEALTH_MAX = 100.0
HEALTH_START = 100.0
HEALTH_PICKUP = 22.0

ENEMY_COUNT = 11
ENEMY_RADIUS = 10
ENEMY_SPEED = 105.0
ENEMY_TOUCH_DAMAGE = 14.0
ENEMY_TOUCH_COOLDOWN = 0.75

JUMP_BUFFER_TIME = 0.10
COYOTE_TIME = 0.12
FLOOR_THEME_CHUNK = 12
WALL_THEME_CHUNK = 8

TAU = math.pi * 2.0
LOADING_TIPS = [
    "TIP: To win, collect 4 boxes, return to the Lobby, and press T.",
    "TIP: Blue items double your speed for 10 seconds.",
    "TIP: Enemies punish slow routes. Keep moving and use open corridors.",
]

GAME_STATES = {"name", "menu", "customization", "loading", "game", "win", "lose"}

PLAYER_COLORS = [
    (255, 30, 30),
    (60, 170, 255),
    (72, 210, 108),
    (248, 174, 46),
    (200, 92, 255),
]


@dataclass
class Camera:
    x: float
    y: float


@dataclass
class PlayerState:
    x: float
    y: float
    z: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    facing: float = 0.0
    grounded: bool = True
    jump_buffer: float = 0.0
    coyote_time: float = 0.0
    health: float = HEALTH_START
    hurt_timer: float = 0.0
    speed_timer: float = 0.0
    color: Tuple[int, int, int] = (255, 30, 30)


@dataclass
class BulletState:
    x: float
    y: float
    z: float
    vx: float
    vy: float
    life: float
    angle: float


@dataclass
class PickupState:
    x: float
    y: float
    sprite_index: int
    bob_phase: float
    heal: float = HEALTH_PICKUP


@dataclass
class BoxState:
    x: float
    y: float
    sprite_index: int
    bob_phase: float
    chunk_x: int
    chunk_y: int
    collected: bool = False


@dataclass
class SpeedItemState:
    x: float
    y: float
    bob_phase: float
    chunk_x: int
    chunk_y: int
    collected: bool = False


@dataclass
class EnemyState:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    wander_angle: float = 0.0
    wander_timer: float = 0.0
    contact_cooldown: float = 0.0


@dataclass
class TileLibrary:
    floor: List[pygame.Surface]
    route: List[pygame.Surface]
    rooms: List[pygame.Surface]
    box: List[pygame.Surface]
    wall: List[pygame.Surface]
    wall_by_role: dict
    pickup: List[pygame.Surface]
    floor_fallback: pygame.Surface
    route_fallback: pygame.Surface
    box_fallback: pygame.Surface
    wall_fallback: pygame.Surface
    pickup_fallback: pygame.Surface


@dataclass
class AssetSemantics:
    room_bias: float
    route_bias: float
    pickup_bias: float
    wall_bias: float
    filename_entropy: float


@dataclass
class ChunkedWorld:
    tiles: TileLibrary
    semantics: AssetSemantics
    generation_seed: int = 0
    chunk_size: int = 24
    chunk_count: int = 48
    start_chunk_x: int = 24
    start_chunk_y: int = 24
    chunks: dict = None
    pickups: List[PickupState] = None
    boxes: List[BoxState] = None
    speed_items: List[SpeedItemState] = None
    enemies: List[EnemyState] = None
    bullets: List[BulletState] = None
    area_names: dict = None
    boxes_collected: int = 0
    lobby_returned: bool = False
    time_left: float = GAME_TIME_LIMIT
    game_over_reason: str = ""
    elapsed_since_chunk: float = 0.0
    travel_progress: float = 0.0
    last_player_chunk: Tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.chunks is None:
            self.chunks = {}
        if self.pickups is None:
            self.pickups = []
        if self.boxes is None:
            self.boxes = []
        if self.speed_items is None:
            self.speed_items = []
        if self.enemies is None:
            self.enemies = []
        if self.bullets is None:
            self.bullets = []
        if self.area_names is None:
            self.area_names = {}

    @property
    def world_size_tiles(self) -> int:
        return self.chunk_count * self.chunk_size

    @property
    def spawn_tile_x(self) -> int:
        return self.start_chunk_x * self.chunk_size + self.chunk_size // 2

    @property
    def spawn_tile_y(self) -> int:
        return self.start_chunk_y * self.chunk_size + self.chunk_size // 2

    @property
    def spawn_world_x(self) -> float:
        return self.spawn_tile_x * TILE_SIZE + TILE_SIZE / 2

    @property
    def spawn_world_y(self) -> float:
        return self.spawn_tile_y * TILE_SIZE + TILE_SIZE / 2

    def clamp_chunk(self, chunk_x: int, chunk_y: int) -> Tuple[int, int]:
        return (
            clamp_int(chunk_x, 0, self.chunk_count - 1),
            clamp_int(chunk_y, 0, self.chunk_count - 1),
        )

    def tile_to_chunk(self, tile_x: int, tile_y: int) -> Tuple[int, int]:
        return tile_x // self.chunk_size, tile_y // self.chunk_size

    def chunk_origin(self, chunk_x: int, chunk_y: int) -> Tuple[int, int]:
        return chunk_x * self.chunk_size, chunk_y * self.chunk_size

    def world_bounds_px(self) -> Tuple[int, int]:
        return self.world_size_tiles * TILE_SIZE, self.world_size_tiles * TILE_SIZE

    def tile_at(self, tile_x: int, tile_y: int) -> int:
        if tile_x < 0 or tile_y < 0 or tile_x >= self.world_size_tiles or tile_y >= self.world_size_tiles:
            return 1
        chunk_x, chunk_y = self.tile_to_chunk(tile_x, tile_y)
        chunk_key = (chunk_x, chunk_y)
        if chunk_key not in self.chunks:
            grid = build_chunk_grid(chunk_x, chunk_y, self)
            self.chunks[chunk_key] = grid
            spawn_entities_for_chunk(self, chunk_x, chunk_y, grid)
            spawn_box_for_chunk(self, chunk_x, chunk_y, grid)
            spawn_speed_item_for_chunk(self, chunk_x, chunk_y, grid)
            self.area_names.setdefault(chunk_key, chunk_area_name(chunk_x, chunk_y, self))
        local_x = tile_x % self.chunk_size
        local_y = tile_y % self.chunk_size
        return int(self.chunks[chunk_key][local_y][local_x])

    def area_name_at(self, tile_x: int, tile_y: int) -> str:
        chunk_x, chunk_y = self.tile_to_chunk(tile_x, tile_y)
        return self.area_names.get((chunk_x, chunk_y), chunk_area_name(chunk_x, chunk_y, self))

    def box_targets(self) -> List[Tuple[int, int]]:
        return [
            (max(0, self.start_chunk_x - 1), self.start_chunk_y),
            (min(self.chunk_count - 1, self.start_chunk_x + 1), self.start_chunk_y),
            (self.start_chunk_x, max(0, self.start_chunk_y - 1)),
            (self.start_chunk_x, min(self.chunk_count - 1, self.start_chunk_y + 1)),
        ]


@dataclass
class WorldBuildConfig:
    enabled: bool
    api_key: str
    base_url: str
    model: str
    image_dir: Path
    cache_path: Path
    timeout_seconds: float
    max_rounds: int
    image_limit: int
    python_recommended: str


@dataclass
class WorldStreamConfig:
    chunk_size: int
    chunk_count: int
    preload_radius: int
    travel_seconds: float

def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_json_file(path: Path, default: dict) -> dict:
    if not path.exists():
        return dict(default)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            merged = dict(default)
            merged.update(data)
            return merged
    except Exception:
        pass
    return dict(default)


def save_json_file(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"Failed to save profile: {exc}")


def load_profile() -> dict:
    defaults = {
        "name": "",
        "color_index": 0,
        "use_player_image": False,
        "time_limit": GAME_TIME_LIMIT,
        "enemy_multiplier": 1.0,
        "show_area_names": True,
    }
    profile = load_json_file(PROFILE_PATH, defaults)
    profile["name"] = str(profile.get("name", "")).strip()
    profile["color_index"] = int(profile.get("color_index", 0)) % len(PLAYER_COLORS)
    profile["use_player_image"] = bool(profile.get("use_player_image", False))
    profile["time_limit"] = float(profile.get("time_limit", GAME_TIME_LIMIT))
    profile["enemy_multiplier"] = float(profile.get("enemy_multiplier", 1.0))
    profile["show_area_names"] = bool(profile.get("show_area_names", True))
    return profile


def save_profile(profile: dict) -> None:
    save_json_file(PROFILE_PATH, profile)


def load_optional_surface(path: Path, size: Tuple[int, int] | None = None) -> pygame.Surface | None:
    if not path.exists():
        return None
    try:
        surface = pygame.image.load(str(path)).convert_alpha()
        if size is not None and surface.get_size() != size:
            surface = pygame.transform.smoothscale(surface, size)
        return surface
    except Exception:
        return None


def scale_cover(surface: pygame.Surface, size: Tuple[int, int]) -> pygame.Surface:
    if surface.get_size() == size:
        return surface
    return pygame.transform.smoothscale(surface, size)


def draw_text(surface: pygame.Surface, font: pygame.font.Font, text: str, color: Tuple[int, int, int], pos: Tuple[int, int]) -> pygame.Rect:
    rendered = font.render(text, True, color)
    rect = rendered.get_rect(topleft=pos)
    surface.blit(rendered, rect)
    return rect


def draw_centered(surface: pygame.Surface, font: pygame.font.Font, text: str, color: Tuple[int, int, int], center: Tuple[int, int]) -> pygame.Rect:
    rendered = font.render(text, True, color)
    rect = rendered.get_rect(center=center)
    surface.blit(rendered, rect)
    return rect


def draw_button(surface: pygame.Surface, rect: pygame.Rect, label: str, font: pygame.font.Font, mouse_pos: Tuple[int, int], accent: Tuple[int, int, int]) -> None:
    hovered = rect.collidepoint(mouse_pos)
    fill = (24, 32, 42) if not hovered else (40, 56, 76)
    outline = accent if hovered else (120, 140, 160)
    pygame.draw.rect(surface, fill, rect, border_radius=8)
    pygame.draw.rect(surface, outline, rect, 2, border_radius=8)
    draw_centered(surface, font, label, (245, 248, 252), rect.center)


def draw_image_button(
    surface: pygame.Surface,
    image: pygame.Surface | None,
    rect: pygame.Rect,
    mouse_pos: Tuple[int, int],
    fallback_label: str,
    font: pygame.font.Font,
    accent: Tuple[int, int, int],
) -> None:
    if image is None:
        draw_button(surface, rect, fallback_label, font, mouse_pos, accent)
        return

    hovered = rect.collidepoint(mouse_pos)
    if hovered:
        glow = pygame.Surface((rect.width + 18, rect.height + 18), pygame.SRCALPHA)
        pygame.draw.rect(glow, (80, 160, 255, 90), glow.get_rect(), border_radius=18)
        surface.blit(glow, glow.get_rect(center=rect.center))

    fitted = pygame.transform.smoothscale(image, rect.size)
    surface.blit(fitted, rect)
    if hovered:
        overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
        overlay.fill((255, 255, 255, 22))
        surface.blit(overlay, rect)


def build_vignette_overlay(size: Tuple[int, int]) -> pygame.Surface:
    w, h = size
    small_w = 64
    small_h = 64
    small = pygame.Surface((small_w, small_h), pygame.SRCALPHA)
    cx = (small_w - 1) / 2.0
    cy = (small_h - 1) / 2.0
    max_dist = math.sqrt(cx * cx + cy * cy)
    for y in range(small_h):
        for x in range(small_w):
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_dist
            alpha = int(clamp((dist - 0.22) / 0.78, 0.0, 1.0) * 210)
            small.set_at((x, y), (0, 0, 0, alpha))
    return pygame.transform.smoothscale(small, (w, h))


def format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def natural_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.stem)
    key: List[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def normalize_grid(grid: Sequence[Sequence[int]], size: int = WORLD_SIZE) -> List[List[int]]:
    normalized = [[1 for _ in range(size)] for _ in range(size)]
    if not isinstance(grid, Sequence):
        return normalized

    for y in range(min(size, len(grid))):
        row = grid[y] if isinstance(grid[y], Sequence) else []
        for x in range(min(size, len(row))):
            try:
                normalized[y][x] = max(0, min(2, int(row[x])))
            except Exception:
                normalized[y][x] = 1

    open_spawn_area(normalized)
    ensure_border_walls(normalized)
    return normalized


def open_spawn_area(grid: List[List[int]]) -> None:
    cx = len(grid[0]) // 2
    cy = len(grid) // 2
    for y in range(cy - 3, cy + 4):
        for x in range(cx - 3, cx + 4):
            if 0 <= x < len(grid[0]) and 0 <= y < len(grid):
                grid[y][x] = 0


def ensure_border_walls(grid: List[List[int]]) -> None:
    width = len(grid[0])
    height = len(grid)
    for x in range(width):
        grid[0][x] = 1
        grid[height - 1][x] = 1
    for y in range(height):
        grid[y][0] = 1
        grid[y][width - 1] = 1


def count_walkable(grid: Sequence[Sequence[int]]) -> int:
    return sum(1 for row in grid for cell in row if int(cell) != 1)


def flood_fill_count(grid: Sequence[Sequence[int]], start_x: int, start_y: int) -> int:
    if not (0 <= start_x < len(grid[0]) and 0 <= start_y < len(grid)):
        return 0
    if int(grid[start_y][start_x]) != 0:
        return 0

    queue = deque([(start_x, start_y)])
    seen = {(start_x, start_y)}
    total = 0
    width = len(grid[0])
    height = len(grid)

    while queue:
        x, y = queue.popleft()
        total += 1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            ny = y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in seen and int(grid[ny][nx]) != 1:
                seen.add((nx, ny))
                queue.append((nx, ny))
    return total


def validate_grid(grid: Sequence[Sequence[int]]) -> bool:
    if len(grid) != WORLD_SIZE or any(len(row) != WORLD_SIZE for row in grid):
        return False

    cx = WORLD_SIZE // 2
    cy = WORLD_SIZE // 2
    if int(grid[cy][cx]) != 0:
        return False

    walkable = count_walkable(grid)
    if walkable < WORLD_SIZE * WORLD_SIZE * 0.18:
        return False
    if walkable > WORLD_SIZE * WORLD_SIZE * 0.70:
        return False

    reachable = flood_fill_count(grid, cx, cy)
    if walkable == 0 or reachable / walkable < 0.82:
        return False

    return True


def carve_rect(grid: List[List[int]], x: int, y: int, w: int, h: int) -> None:
    for yy in range(max(1, y), min(len(grid) - 1, y + h)):
        for xx in range(max(1, x), min(len(grid[0]) - 1, x + w)):
            grid[yy][xx] = 0


def carve_h_corridor(grid: List[List[int]], x1: int, x2: int, y: int, radius: int = 1, tile_value: int = 2) -> None:
    left = min(x1, x2)
    right = max(x1, x2)
    for yy in range(max(1, y - radius), min(len(grid) - 1, y + radius + 1)):
        for xx in range(max(1, left), min(len(grid[0]) - 1, right + 1)):
            grid[yy][xx] = tile_value


def carve_v_corridor(grid: List[List[int]], y1: int, y2: int, x: int, radius: int = 1, tile_value: int = 2) -> None:
    top = min(y1, y2)
    bottom = max(y1, y2)
    for yy in range(max(1, top), min(len(grid) - 1, bottom + 1)):
        for xx in range(max(1, x - radius), min(len(grid[0]) - 1, x + radius + 1)):
            grid[yy][xx] = tile_value


def rooms_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], padding: int = 2) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (
        ax + aw + padding < bx
        or bx + bw + padding < ax
        or ay + ah + padding < by
        or by + bh + padding < ay
    )


def room_center(room: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x, y, w, h = room
    return x + w // 2, y + h // 2


def connect_points(grid: List[List[int]], a: Tuple[int, int], b: Tuple[int, int], rng: random.Random) -> None:
    ax, ay = a
    bx, by = b
    if rng.random() < 0.5:
        carve_h_corridor(grid, ax, bx, ay, radius=1, tile_value=2)
        carve_v_corridor(grid, ay, by, bx, radius=1, tile_value=2)
    else:
        carve_v_corridor(grid, ay, by, ax, radius=1, tile_value=2)
        carve_h_corridor(grid, ax, bx, by, radius=1, tile_value=2)


def generate_dungeon_map(size: int = WORLD_SIZE, seed: int | None = None) -> List[List[int]]:
    grid = [[1 for _ in range(size)] for _ in range(size)]

    def add_room(x: int, y: int, w: int, h: int) -> None:
        carve_rect(grid, x, y, w, h)

    def add_route_h(x1: int, x2: int, y: int, radius: int = 1) -> None:
        carve_h_corridor(grid, x1, x2, y, radius=radius, tile_value=2)

    def add_route_v(y1: int, y2: int, x: int, radius: int = 1) -> None:
        carve_v_corridor(grid, y1, y2, x, radius=radius, tile_value=2)

    # Main layout: central hall + four wings + corner annexes.
    add_room(22, 22, 20, 20)
    add_room(6, 8, 12, 10)
    add_room(46, 8, 12, 10)
    add_room(6, 46, 12, 10)
    add_room(46, 46, 12, 10)
    add_room(18, 6, 28, 10)
    add_room(18, 48, 28, 10)
    add_room(6, 18, 10, 28)
    add_room(48, 18, 10, 28)

    # Corridors that connect everything cleanly.
    add_route_h(18, 22, 13, radius=1)
    add_route_h(40, 46, 13, radius=1)
    add_route_h(18, 22, 51, radius=1)
    add_route_h(40, 46, 51, radius=1)
    add_route_v(13, 22, 13, radius=1)
    add_route_v(13, 22, 51, radius=1)
    add_route_v(40, 51, 13, radius=1)
    add_route_v(40, 51, 51, radius=1)

    add_route_h(12, 52, 32, radius=1)
    add_route_v(12, 52, 32, radius=1)

    # Small internal rooms to break the monotony.
    add_room(26, 8, 12, 8)
    add_room(26, 48, 12, 8)
    add_room(8, 26, 8, 12)
    add_room(48, 26, 8, 12)

    open_spawn_area(grid)
    ensure_border_walls(grid)
    return grid


def generate_random_fallback_map(size: int = WORLD_SIZE) -> List[List[int]]:
    return generate_dungeon_map(size)


def analyze_asset_semantics(base_dir: Path = Path("uses")) -> AssetSemantics:
    folder_weights = {
        "Rooms": 0.0,
        "Route": 0.0,
        "Pickup": 0.0,
        "Wall": 0.0,
    }
    numeric_total = 0
    token_total = 0
    file_total = 0

    for folder_name in folder_weights:
        folder = base_dir / folder_name
        if not folder.exists() or not folder.is_dir():
            continue
        files = [path for path in folder.iterdir() if path.is_file()]
        file_total += len(files)
        folder_weights[folder_name] = float(len(files))
        for path in files:
            tokens = re.findall(r"[A-Za-z]+|\d+", path.stem)
            token_total += len(tokens)
            numeric_total += sum(int(token) for token in tokens if token.isdigit())

    scale = max(1.0, float(file_total))
    return AssetSemantics(
        room_bias=0.8 + folder_weights["Rooms"] / scale,
        route_bias=0.8 + folder_weights["Route"] / scale,
        pickup_bias=0.6 + folder_weights["Pickup"] / scale,
        wall_bias=0.9 + folder_weights["Wall"] / scale,
        filename_entropy=min(2.0, (token_total + numeric_total * 0.02) / max(1.0, scale * 3.0)),
    )


def chunk_seed(chunk_x: int, chunk_y: int, salt: int = 0) -> int:
    return abs((chunk_x * 73856093) ^ (chunk_y * 19349663) ^ salt) & 0xFFFFFFFF


def world_chunk_seed(world: ChunkedWorld, chunk_x: int, chunk_y: int, salt: int = 0) -> int:
    return chunk_seed(chunk_x, chunk_y, salt ^ world.generation_seed)


def chunk_role(chunk_x: int, chunk_y: int, world: ChunkedWorld) -> str:
    room_bias = world.semantics.room_bias
    route_bias = world.semantics.route_bias
    role_roll = (world_chunk_seed(world, chunk_x, chunk_y, 0x5EED) % 100) / 100.0
    if chunk_x == world.start_chunk_x and chunk_y == world.start_chunk_y:
        return "hub"
    if route_bias > room_bias and role_roll < 0.5:
        return "route"
    if room_bias > route_bias and role_roll < 0.7:
        return "room"
    return "mixed"


def chunk_area_name(chunk_x: int, chunk_y: int, world: ChunkedWorld) -> str:
    if chunk_x == world.start_chunk_x and chunk_y == world.start_chunk_y:
        return "Lobby"

    role = chunk_role(chunk_x, chunk_y, world)
    prefixes = {
        "room": ["Quartz", "Sunlit", "Amber", "Archive", "Silent", "North", "West", "South", "East"],
        "route": ["Echo", "Drift", "Copper", "Thread", "Ribbon", "Pass", "Link", "Trace"],
        "mixed": ["Bridge", "Cavern", "Hall", "Crossing", "Node", "Gate", "Nook", "Span"],
        "hub": ["Lobby", "Atrium", "Center", "Core"],
    }
    suffixes = {
        "room": ["Room", "Wing", "Chamber", "Gallery", "Vault"],
        "route": ["Corridor", "Way", "Run", "Passage", "Line"],
        "mixed": ["Hall", "Section", "Node", "Sector"],
        "hub": ["Lobby"],
    }
    pool_a = prefixes.get(role, prefixes["mixed"])
    pool_b = suffixes.get(role, suffixes["mixed"])
    seed = world_chunk_seed(world, chunk_x, chunk_y, 0xCAFE)
    a = pool_a[seed % len(pool_a)]
    b = pool_b[(seed // 7) % len(pool_b)]
    if role == "route":
        return f"{a} {b}"
    if role == "room":
        return f"{a} {b}"
    if role == "mixed":
        return f"{a} {b}"
    return "Lobby"


def edge_open(chunk_x: int, chunk_y: int, side: str, world: ChunkedWorld) -> bool:
    if side == "right":
        a = (chunk_x, chunk_y)
        b = (chunk_x + 1, chunk_y)
        salt = 17
    elif side == "left":
        a = (chunk_x - 1, chunk_y)
        b = (chunk_x, chunk_y)
        salt = 17
    elif side == "down":
        a = (chunk_x, chunk_y)
        b = (chunk_x, chunk_y + 1)
        salt = 31
    else:
        a = (chunk_x, chunk_y - 1)
        b = (chunk_x, chunk_y)
        salt = 31

    if world.start_chunk_x in {a[0], b[0]} and world.start_chunk_y in {a[1], b[1]}:
        return True

    pair_a, pair_b = sorted((a, b))
    value = world_chunk_seed(world, pair_a[0] ^ pair_b[0], pair_a[1] ^ pair_b[1], salt)
    base_threshold = 54 if world.semantics.route_bias >= world.semantics.room_bias else 42
    threshold = base_threshold + int(world.semantics.filename_entropy * 6)
    return (value % 100) < threshold


def ensure_chunk_opening(grid: List[List[int]], side: str) -> None:
    size = len(grid)
    mid = size // 2
    span = max(1, size // 8)
    if side == "left":
        for y in range(mid - span, mid + span + 1):
            grid[y][0] = 0
    elif side == "right":
        for y in range(mid - span, mid + span + 1):
            grid[y][size - 1] = 0
    elif side == "up":
        for x in range(mid - span, mid + span + 1):
            grid[0][x] = 0
    elif side == "down":
        for x in range(mid - span, mid + span + 1):
            grid[size - 1][x] = 0


def carve_hub_connections(grid: List[List[int]]) -> None:
    size = len(grid)
    mid = size // 2
    inner = max(2, size // 3)
    carve_rect(grid, mid - 2, mid - 2, 5, 5)
    carve_h_corridor(grid, 1, size - 2, mid, radius=1, tile_value=0)
    carve_v_corridor(grid, 1, size - 2, mid, radius=1, tile_value=0)
    carve_h_corridor(grid, 1, mid, inner, radius=1, tile_value=0)
    carve_h_corridor(grid, mid, size - 2, size - inner - 1, radius=1, tile_value=0)
    carve_v_corridor(grid, 1, mid, inner, radius=1, tile_value=0)
    carve_v_corridor(grid, mid, size - 2, size - inner - 1, radius=1, tile_value=0)
    ensure_chunk_opening(grid, "left")
    ensure_chunk_opening(grid, "right")
    ensure_chunk_opening(grid, "up")
    ensure_chunk_opening(grid, "down")


def connect_chunk_to_edge(grid: List[List[int]], side: str) -> None:
    size = len(grid)
    mid = size // 2
    if side == "left":
        carve_h_corridor(grid, 1, mid, mid, radius=1, tile_value=0)
        carve_rect(grid, 0, mid - 1, 2, 3)
    elif side == "right":
        carve_h_corridor(grid, mid, size - 2, mid, radius=1, tile_value=0)
        carve_rect(grid, size - 2, mid - 1, 2, 3)
    elif side == "up":
        carve_v_corridor(grid, 1, mid, mid, radius=1, tile_value=0)
        carve_rect(grid, mid - 1, 0, 3, 2)
    elif side == "down":
        carve_v_corridor(grid, mid, size - 2, mid, radius=1, tile_value=0)
        carve_rect(grid, mid - 1, size - 2, 3, 2)


def carve_chunk_room(grid: List[List[int]], rng: random.Random, semantics: AssetSemantics) -> None:
    size = len(grid)
    room_w = rng.randint(max(8, size // 2), max(10, size - 4))
    room_h = rng.randint(max(8, size // 2), max(10, size - 4))
    x = rng.randint(1, max(1, size - room_w - 1))
    y = rng.randint(1, max(1, size - room_h - 1))
    carve_rect(grid, x, y, room_w, room_h)

    if semantics.room_bias > 1.0 and rng.random() < 0.7:
        extra_w = max(4, room_w // 2)
        extra_h = max(4, room_h // 2)
        carve_rect(grid, clamp_int(x - 2, 1, size - extra_w - 1), clamp_int(y + room_h // 3, 1, size - extra_h - 1), extra_w, extra_h)


def carve_chunk_route(grid: List[List[int]], rng: random.Random, semantics: AssetSemantics) -> None:
    size = len(grid)
    mid = size // 2
    corridor_radius = 1 if semantics.route_bias < 1.2 else 2
    carve_rect(grid, max(2, mid - 4), 2, min(size - 4, 9), size - 4)
    carve_rect(grid, 2, max(2, mid - 4), size - 4, min(size - 4, 9))
    if rng.random() < 0.5:
        carve_h_corridor(grid, 1, size - 2, mid, radius=corridor_radius, tile_value=2)
        if rng.random() < 0.65:
            carve_v_corridor(grid, 1, size - 2, clamp_int(mid + rng.randint(-3, 3), 2, size - 3), radius=1, tile_value=2)
    else:
        carve_v_corridor(grid, 1, size - 2, mid, radius=corridor_radius, tile_value=2)
        if rng.random() < 0.65:
            carve_h_corridor(grid, 1, size - 2, clamp_int(mid + rng.randint(-3, 3), 2, size - 3), radius=1, tile_value=2)


def build_chunk_grid(chunk_x: int, chunk_y: int, world: ChunkedWorld) -> List[List[int]]:
    size = world.chunk_size
    semantics = world.semantics
    rng = random.Random(world_chunk_seed(world, chunk_x, chunk_y, 0xA11CE))
    grid = [[1 for _ in range(size)] for _ in range(size)]
    room_bias = semantics.room_bias + (rng.random() * 0.35)
    route_bias = semantics.route_bias + (rng.random() * 0.35)
    role = chunk_role(chunk_x, chunk_y, world)

    if role in {"room", "hub", "mixed"}:
        carve_chunk_room(grid, rng, semantics)
    if role in {"route", "hub", "mixed"}:
        carve_chunk_route(grid, rng, semantics)

    if role == "hub":
        carve_hub_connections(grid)

    open_sides = [side for side in ("left", "right", "up", "down") if edge_open(chunk_x, chunk_y, side, world)]
    if role == "hub":
        open_sides = ["left", "right", "up", "down"]
    elif not open_sides:
        preferred = "right" if role == "route" else ("down" if (world_chunk_seed(world, chunk_x, chunk_y, 0xBEEF) % 2 == 0) else "left")
        open_sides = [preferred]
    elif role in {"route", "mixed"} and len(open_sides) < 2:
        for candidate in ("right", "down", "left", "up"):
            if candidate not in open_sides:
                open_sides.append(candidate)
                break

    for side in open_sides:
        ensure_chunk_opening(grid, side)
        connect_chunk_to_edge(grid, side)

    mid = size // 2
    carve_rect(grid, mid - 1, mid - 1, 3, 3)
    if role == "hub":
        carve_hub_connections(grid)
    open_spawn_area(grid)
    return grid


def is_chunk_near_spawn(chunk_x: int, chunk_y: int, world: ChunkedWorld) -> bool:
    return abs(chunk_x - world.start_chunk_x) <= 1 and abs(chunk_y - world.start_chunk_y) <= 1


def spawn_entities_for_chunk(world: ChunkedWorld, chunk_x: int, chunk_y: int, grid: Sequence[Sequence[int]]) -> None:
    rng = random.Random(world_chunk_seed(world, chunk_x, chunk_y, 0xE11F))
    open_tiles: List[Tuple[int, int, int]] = []
    size = len(grid)
    role = chunk_role(chunk_x, chunk_y, world)
    for y in range(1, size - 1):
        for x in range(1, size - 1):
            if int(grid[y][x]) != 0:
                continue
            if abs(x - size // 2) + abs(y - size // 2) < 4:
                continue
            if x < 3 or y < 3 or x > size - 4 or y > size - 4:
                continue
            open_neighbors = count_open_neighbors(grid, x, y)
            if open_neighbors < 3:
                continue
            open_tiles.append((open_neighbors, x, y))

    open_tiles.sort(reverse=True)
    base_x, base_y = world.chunk_origin(chunk_x, chunk_y)
    room_score = 0.0
    route_score = 0.0
    if role in {"room", "hub"}:
        room_score = 1.0
    if role in {"route", "hub"}:
        route_score = 1.0
    if is_chunk_near_spawn(chunk_x, chunk_y, world):
        room_score = 0.0
        route_score = 0.0

    local_pickups = 1 if room_score > 0.0 else 0
    local_enemies = 2 if route_score > 0.0 else 1 if room_score > 0.0 else 0

    for _, x, y in open_tiles[: local_pickups * 2]:
        if rng.random() < 0.25:
            continue
        sprite_index = stable_index(base_x + x, base_y + y, 0xF00D, max(1, len(world.tiles.pickup)))
        world.pickups.append(
            PickupState(
                x=(base_x + x) * TILE_SIZE + TILE_SIZE / 2,
                y=(base_y + y) * TILE_SIZE + TILE_SIZE / 2,
                sprite_index=sprite_index,
                bob_phase=rng.random() * TAU,
                heal=HEALTH_PICKUP,
            )
        )

    for _, x, y in open_tiles[local_pickups : local_pickups + local_enemies * 3]:
        if rng.random() < 0.75:
            continue
        world.enemies.append(
            EnemyState(
                x=(base_x + x) * TILE_SIZE + TILE_SIZE / 2,
                y=(base_y + y) * TILE_SIZE + TILE_SIZE / 2,
                wander_angle=rng.random() * TAU,
                wander_timer=rng.uniform(0.6, 1.8),
                contact_cooldown=rng.uniform(0.0, 0.6),
            )
        )


def spawn_box_for_chunk(world: ChunkedWorld, chunk_x: int, chunk_y: int, grid: Sequence[Sequence[int]]) -> None:
    if (chunk_x, chunk_y) not in world.box_targets():
        return
    if any(box.chunk_x == chunk_x and box.chunk_y == chunk_y for box in world.boxes):
        return

    rng = random.Random(world_chunk_seed(world, chunk_x, chunk_y, 0xB0B))
    size = len(grid)
    candidates: List[Tuple[int, int, int]] = []
    for y in range(2, size - 2):
        for x in range(2, size - 2):
            if int(grid[y][x]) != 0:
                continue
            open_neighbors = count_open_neighbors(grid, x, y)
            if open_neighbors < 3:
                continue
            distance = abs(x - size // 2) + abs(y - size // 2)
            candidates.append((distance * 10 + open_neighbors * 100, x, y))

    candidates.sort(reverse=True)
    if not candidates:
        return

    _, x, y = candidates[0]
    base_x, base_y = world.chunk_origin(chunk_x, chunk_y)
    sprite_index = stable_index(chunk_x ^ world.generation_seed, chunk_y ^ (world.generation_seed >> 1), 0xB0B, max(1, len(world.tiles.box)))
    world.boxes.append(
        BoxState(
            x=(base_x + x) * TILE_SIZE + TILE_SIZE / 2,
            y=(base_y + y) * TILE_SIZE + TILE_SIZE / 2,
            sprite_index=sprite_index,
            bob_phase=rng.random() * TAU,
            chunk_x=chunk_x,
            chunk_y=chunk_y,
        )
    )


def spawn_speed_item_for_chunk(world: ChunkedWorld, chunk_x: int, chunk_y: int, grid: Sequence[Sequence[int]]) -> None:
    if is_chunk_near_spawn(chunk_x, chunk_y, world):
        return
    if any(item.chunk_x == chunk_x and item.chunk_y == chunk_y for item in world.speed_items):
        return
    if world_chunk_seed(world, chunk_x, chunk_y, 0x5AEE) % 100 > 38:
        return

    size = len(grid)
    candidates: List[Tuple[int, int, int]] = []
    for y in range(2, size - 2):
        for x in range(2, size - 2):
            if int(grid[y][x]) != 0:
                continue
            open_neighbors = count_open_neighbors(grid, x, y)
            if open_neighbors < 3:
                continue
            distance = abs(x - size // 2) + abs(y - size // 2)
            candidates.append((open_neighbors * 100 - distance, x, y))

    if not candidates:
        return

    candidates.sort(reverse=True)
    _, x, y = candidates[0]
    base_x, base_y = world.chunk_origin(chunk_x, chunk_y)
    rng = random.Random(world_chunk_seed(world, chunk_x, chunk_y, 0x5AEE))
    world.speed_items.append(
        SpeedItemState(
            x=(base_x + x) * TILE_SIZE + TILE_SIZE / 2,
            y=(base_y + y) * TILE_SIZE + TILE_SIZE / 2,
            bob_phase=rng.random() * TAU,
            chunk_x=chunk_x,
            chunk_y=chunk_y,
        )
    )


def build_streaming_world(
    tiles: TileLibrary,
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    stream_config: WorldStreamConfig,
) -> ChunkedWorld:
    semantics = analyze_asset_semantics()
    start_chunk = stream_config.chunk_count // 2
    world = ChunkedWorld(
        tiles=tiles,
        semantics=semantics,
        generation_seed=random.randint(0, 2**31 - 1),
        chunk_size=stream_config.chunk_size,
        chunk_count=stream_config.chunk_count,
        start_chunk_x=start_chunk,
        start_chunk_y=start_chunk,
    )
    progress = 0.05
    draw_loading_screen(surface, title_font, body_font, progress, "Building World", LOADING_TIPS[0])
    for idx, (chunk_x, chunk_y) in enumerate(
        (
            (world.start_chunk_x, world.start_chunk_y),
            (world.start_chunk_x + 1, world.start_chunk_y),
            (world.start_chunk_x - 1, world.start_chunk_y),
            (world.start_chunk_x, world.start_chunk_y + 1),
            (world.start_chunk_x, world.start_chunk_y - 1),
            (world.start_chunk_x + 1, world.start_chunk_y + 1),
            (world.start_chunk_x - 1, world.start_chunk_y - 1),
            (world.start_chunk_x + 1, world.start_chunk_y - 1),
            (world.start_chunk_x - 1, world.start_chunk_y + 1),
        )
    ):
        chunk_x, chunk_y = world.clamp_chunk(chunk_x, chunk_y)
        if (chunk_x, chunk_y) in world.chunks:
            continue
        tip = LOADING_TIPS[idx % len(LOADING_TIPS)]
        draw_loading_screen(
            surface,
            title_font,
            body_font,
            progress,
            "Building World",
            f"{tip} | Preparing chunk {idx + 1} using local layout semantics",
        )
        grid = build_chunk_grid(chunk_x, chunk_y, world)
        world.chunks[(chunk_x, chunk_y)] = grid
        spawn_entities_for_chunk(world, chunk_x, chunk_y, grid)
        spawn_box_for_chunk(world, chunk_x, chunk_y, grid)
        spawn_speed_item_for_chunk(world, chunk_x, chunk_y, grid)
        world.area_names[(chunk_x, chunk_y)] = chunk_area_name(chunk_x, chunk_y, world)
        progress = min(0.9, progress + 0.08)
    draw_loading_screen(surface, title_font, body_font, 1.0, "Building World", "Streaming world ready")
    return world


def preload_chunks_around(world: ChunkedWorld, center_chunk_x: int, center_chunk_y: int, radius: int) -> None:
    for cy in range(center_chunk_y - radius, center_chunk_y + radius + 1):
        for cx in range(center_chunk_x - radius, center_chunk_x + radius + 1):
            cx, cy = world.clamp_chunk(cx, cy)
            key = (cx, cy)
            if key in world.chunks:
                continue
            grid = build_chunk_grid(cx, cy, world)
            world.chunks[key] = grid
            spawn_entities_for_chunk(world, cx, cy, grid)
            spawn_box_for_chunk(world, cx, cy, grid)
            spawn_speed_item_for_chunk(world, cx, cy, grid)
            world.area_names[key] = chunk_area_name(cx, cy, world)


def update_streaming_world(world: ChunkedWorld, player: PlayerState, dt: float, stream_config: WorldStreamConfig) -> None:
    current_chunk_x, current_chunk_y = world.tile_to_chunk(int(player.x // TILE_SIZE), int(player.y // TILE_SIZE))
    current_chunk_x, current_chunk_y = world.clamp_chunk(current_chunk_x, current_chunk_y)
    if world.last_player_chunk != (current_chunk_x, current_chunk_y):
        world.last_player_chunk = (current_chunk_x, current_chunk_y)
        preload_chunks_around(world, current_chunk_x, current_chunk_y, stream_config.preload_radius)

    if player.vx != 0.0 or player.vy != 0.0:
        world.travel_progress += dt
    else:
        world.travel_progress = max(0.0, world.travel_progress - dt * 0.35)

    if world.travel_progress >= stream_config.travel_seconds:
        world.travel_progress = 0.0
        preload_chunks_around(world, current_chunk_x, current_chunk_y, stream_config.preload_radius + 1)


def get_env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)).strip())
    except Exception:
        return default


def load_world_build_config() -> WorldBuildConfig:
    return WorldBuildConfig(
        enabled=get_env_bool("WORLD_AI_ENABLED", True),
        api_key=(os.getenv("NVIDIA_NIM_API_KEY", "") or os.getenv("NVIDIA_API_KEY", "")).strip(),
        base_url=os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1").strip().rstrip("/"),
        model=os.getenv("NVIDIA_NIM_MODEL", "google/gemma-4-31B-it").strip(),
        image_dir=Path(os.getenv("WORLD_AI_IMAGE_DIR", "img").strip() or "img"),
        cache_path=Path(os.getenv("WORLD_AI_CACHE_PATH", "uses/generated_world.json").strip() or "uses/generated_world.json"),
        timeout_seconds=get_env_float("WORLD_AI_TIMEOUT", 90.0),
        max_rounds=max(1, get_env_int("WORLD_AI_MAX_ROUNDS", 3)),
        image_limit=max(1, get_env_int("WORLD_AI_IMAGE_LIMIT", 4)),
        python_recommended=os.getenv("PYTHON_RECOMMENDED_VERSION", "3.10").strip() or "3.10",
    )


def load_world_stream_config() -> WorldStreamConfig:
    return WorldStreamConfig(
        chunk_size=max(12, get_env_int("WORLD_CHUNK_SIZE", 24)),
        chunk_count=max(8, get_env_int("WORLD_CHUNK_COUNT", 48)),
        preload_radius=max(1, get_env_int("WORLD_STREAM_PRELOAD_RADIUS", 1)),
        travel_seconds=max(15.0, get_env_float("WORLD_STREAM_TRAVEL_SECONDS", 120.0)),
    )


def collect_reference_images(image_dir: Path, image_limit: int) -> List[Path]:
    candidates: List[Path] = []
    search_dirs = [image_dir, Path(".")]
    suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    preferred_tokens = ("sheet", "candidate", "selected")

    for folder in search_dirs:
        if not folder.exists() or not folder.is_dir():
            continue
        for path in folder.iterdir():
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            lowered = path.name.lower()
            if folder == image_dir or any(token in lowered for token in preferred_tokens):
                candidates.append(path)

    unique: List[Path] = []
    seen = set()
    for path in sorted(candidates, key=lambda item: (item.parent.as_posix().lower(), natural_key(item))):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
        if len(unique) >= image_limit:
            break
    return unique


def path_to_data_url(path: Path) -> str:
    surface = pygame.image.load(str(path))
    width, height = surface.get_size()
    max_side = 384
    scale = min(1.0, max_side / max(1, width), max_side / max(1, height))
    if scale < 1.0:
        surface = pygame.transform.smoothscale(surface, (max(1, int(width * scale)), max(1, int(height * scale))))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = Path(tmp.name)

    try:
        pygame.image.save(surface, str(temp_path))
        encoded = base64.b64encode(temp_path.read_bytes()).decode("ascii")
    finally:
        try:
            temp_path.unlink()
        except Exception:
            pass

    return f"data:image/png;base64,{encoded}"


def http_post_json(url: str, payload: dict, api_key: str, timeout_seconds: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response_text = response.read().decode("utf-8", errors="replace")
    return json.loads(response_text)


def http_post_json_with_errors(url: str, payload: dict, api_key: str, timeout_seconds: float) -> dict:
    try:
        return http_post_json(url, payload, api_key, timeout_seconds)
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {body}".strip()) from exc


def extract_json_grid(text: str) -> List[List[int]] | None:
    candidates = []
    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    candidates.append(text)

    for candidate in candidates:
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start == -1 or end == -1 or end <= start:
            continue
        snippet = candidate[start : end + 1]
        try:
            parsed = json.loads(snippet)
        except Exception:
            continue
        if isinstance(parsed, dict) and "grid" in parsed:
            parsed = parsed["grid"]
        if isinstance(parsed, list):
            return parsed  # normalized later
    return None


def score_grid(grid: Sequence[Sequence[int]]) -> float:
    if len(grid) != WORLD_SIZE or any(len(row) != WORLD_SIZE for row in grid):
        return 0.0

    walkable = count_walkable(grid)
    total_cells = WORLD_SIZE * WORLD_SIZE
    walkable_ratio = walkable / total_cells if total_cells else 0.0
    reachable = flood_fill_count(grid, WORLD_SIZE // 2, WORLD_SIZE // 2)
    connectivity = reachable / walkable if walkable else 0.0
    route_cells = sum(1 for row in grid for cell in row if int(cell) == 2)
    route_ratio = route_cells / total_cells if total_cells else 0.0

    # Bias toward compact, traversable layouts with some connective tissue.
    score = 0.0
    score += 1.0 - min(1.0, abs(walkable_ratio - 0.42) / 0.42)
    score += min(1.0, connectivity)
    score += min(1.0, route_ratio / 0.12 if route_ratio else 0.0)
    return score / 3.0


def merge_ai_grid(base_grid: Sequence[Sequence[int]], ai_grid: Sequence[Sequence[int]]) -> List[List[int]]:
    merged = normalize_grid(base_grid)
    if len(ai_grid) != WORLD_SIZE or any(len(row) != WORLD_SIZE for row in ai_grid):
        return merged

    for y in range(WORLD_SIZE):
        for x in range(WORLD_SIZE):
            try:
                cell = int(ai_grid[y][x])
            except Exception:
                continue
            if cell in (0, 2):
                merged[y][x] = cell

    open_spawn_area(merged)
    ensure_border_walls(merged)
    return merged


def save_world_cache(cache_path: Path, grid: Sequence[Sequence[int]]) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"grid": grid}, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"Failed to save world cache: {exc}")


def ask_nim_for_map(
    config: WorldBuildConfig,
    reference_images: Sequence[Path],
    round_index: int,
    previous_feedback: str | None = None,
) -> List[List[int]] | None:
    if not config.api_key:
        return None

    image_lines = "\n".join(f"- {path.name}" for path in reference_images) if reference_images else "- none"
    image_parts = []
    for path in reference_images:
        image_parts.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": path_to_data_url(path),
                },
            }
        )

    prompt = (
        "Build a 64x64 top-down dungeon map for a 2.5D pygame game.\n"
        "Tile meanings:\n"
        "0 = floor/open space\n"
        "1 = wall/blocked space\n"
        "2 = route/corridor floor\n"
        "Rules:\n"
        "- Return only valid JSON.\n"
        "- The JSON must contain a single key named grid.\n"
        "- grid must be a 64x64 nested array of integers.\n"
        "- The center tile must be open floor.\n"
        "- The map must be fully playable and mostly connected.\n"
        "- Use the attached images as the source of visual layout guidance.\n"
    )
    if previous_feedback:
        prompt += f"\nPrevious attempt feedback: {previous_feedback}\n"
    prompt += f"\nRefinement round: {round_index + 1}.\nAttached images:\n{image_lines}\n"

    payload = {
        "model": config.model,
        "temperature": 0.2,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "system",
                "content": "You are a map-building assistant that only emits valid JSON.",
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}] + image_parts,
            },
        ],
    }

    try:
        response = http_post_json_with_errors(f"{config.base_url}/chat/completions", payload, config.api_key, config.timeout_seconds)
    except Exception as exc:
        print(f"AI map build request failed: {exc}")
        return None

    try:
        content = response["choices"][0]["message"]["content"]
    except Exception:
        content = ""

    parsed = extract_json_grid(content)
    if parsed is None:
        return None

    candidate = normalize_grid(parsed)
    if not validate_grid(candidate):
        return None
    return candidate


def build_world_with_ai(
    config: WorldBuildConfig,
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
) -> List[List[int]]:
    def progress(stage: float, headline: str, detail: str) -> None:
        draw_loading_screen(surface, title_font, body_font, stage, headline, detail)

    progress(0.05, "Building World", f"Python {sys.version_info.major}.{sys.version_info.minor} loaded")

    if not config.enabled:
        progress(1.0, "Building World", "AI world builder disabled, using fallback dungeon")
        return generate_dungeon_map(WORLD_SIZE)

    if not config.api_key:
        progress(0.35, "Building World", "NVIDIA_NIM_API_KEY is empty, using fallback dungeon")
        return generate_dungeon_map(WORLD_SIZE)

    if config.python_recommended != f"{sys.version_info.major}.{sys.version_info.minor}":
        print(f"Recommended Python version: {config.python_recommended}, current: {sys.version_info.major}.{sys.version_info.minor}")

    reference_images = collect_reference_images(config.image_dir, config.image_limit)
    if not reference_images:
        progress(0.35, "Building World", "No reference images found, using fallback dungeon")
        return generate_dungeon_map(WORLD_SIZE)

    progress(0.20, "Building World", f"Found {len(reference_images)} reference images")

    fallback_grid = generate_dungeon_map(WORLD_SIZE)
    best_grid = fallback_grid
    best_score = score_grid(fallback_grid)
    previous_feedback: str | None = None

    for round_index in range(config.max_rounds):
        progress(
            0.30 + (0.55 * round_index / max(1, config.max_rounds)),
            "Building World",
            f"Gemma refinement round {round_index + 1} of {config.max_rounds}",
        )
        ai_grid = ask_nim_for_map(config, reference_images, round_index, previous_feedback=previous_feedback)
        if ai_grid is None:
            previous_feedback = "The model response did not contain a valid 64x64 JSON grid."
            break

        merged = merge_ai_grid(fallback_grid, ai_grid)
        current_score = score_grid(merged)
        if validate_grid(merged) and current_score >= best_score:
            progress(0.95, "Building World", "World accepted from AI output")
            save_world_cache(config.cache_path, merged)
            return merged

        best_grid = merged if current_score > best_score else best_grid
        best_score = max(best_score, current_score)
        previous_feedback = (
            f"Improve connectivity and playability. Current reward score is {current_score:.3f}. "
            "Keep the center open and ensure the map stays connected."
        )

    progress(0.90, "Building World", "Using best repaired layout after refinement")
    if validate_grid(best_grid):
        save_world_cache(config.cache_path, best_grid)
        return best_grid
    return fallback_grid


# ---------------------------
# Math / collision helpers
# ---------------------------
def normalize(vx: float, vy: float) -> Tuple[float, float]:
    length = math.hypot(vx, vy)
    if length == 0:
        return 0.0, 0.0
    return vx / length, vy / length


def approach(current: float, target: float, delta: float) -> float:
    if current < target:
        return min(current + delta, target)
    if current > target:
        return max(current - delta, target)
    return target


def rect_from_center(cx: float, cy: float, w: float, h: float) -> pygame.Rect:
    return pygame.Rect(int(cx - w / 2), int(cy - h / 2), int(w), int(h))


def tile_center(tx: int, ty: int) -> Tuple[float, float]:
    return tx * TILE_SIZE + TILE_SIZE / 2, ty * TILE_SIZE + TILE_SIZE / 2


def world_to_screen(camera: Camera, world_x: float, world_y: float, world_z: float = 0.0) -> Tuple[int, int]:
    screen_x = int(world_x - camera.x + VIEW_WIDTH / 2)
    screen_y = int(world_y - camera.y + VIEW_HEIGHT / 2 - world_z)
    return screen_x, screen_y


def count_open_neighbors(grid: Sequence[Sequence[int]] | ChunkedWorld, x: int, y: int) -> int:
    total = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx = x + dx
        ny = y + dy
        if isinstance(grid, ChunkedWorld):
            if int(grid.tile_at(nx, ny)) == 0:
                total += 1
        elif 0 <= nx < len(grid[0]) and 0 <= ny < len(grid) and int(grid[ny][nx]) == 0:
            total += 1
    return total


def stable_index(x: int, y: int, salt: int, count: int) -> int:
    if count <= 0:
        return 0
    value = (x * 73856093) ^ (y * 19349663) ^ salt
    return abs(value) % count


def rect_hits_walls(rect: pygame.Rect, grid: Sequence[Sequence[int]] | ChunkedWorld) -> bool:
    min_x = rect.left // TILE_SIZE
    max_x = (rect.right - 1) // TILE_SIZE
    min_y = rect.top // TILE_SIZE
    max_y = (rect.bottom - 1) // TILE_SIZE

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            cell = grid.tile_at(x, y) if isinstance(grid, ChunkedWorld) else int(grid[y][x]) if 0 <= y < len(grid) and 0 <= x < len(grid[0]) else 1
            if int(cell) == 1:
                wall_rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                if rect.colliderect(wall_rect):
                    return True
    return False


def resolve_player_axis(player: PlayerState, grid: Sequence[Sequence[int]] | ChunkedWorld, dx: float, dy: float) -> None:
    if dx != 0.0:
        player.x += dx
        rect = rect_from_center(player.x, player.y, PLAYER_RADIUS * 2, PLAYER_RADIUS * 2)
        if rect_hits_walls(rect, grid):
            player.x -= dx
            player.vx = 0.0

    if dy != 0.0:
        player.y += dy
        rect = rect_from_center(player.x, player.y, PLAYER_RADIUS * 2, PLAYER_RADIUS * 2)
        if rect_hits_walls(rect, grid):
            player.y -= dy
            player.vy = 0.0


def build_pickups(grid: Sequence[Sequence[int]] | ChunkedWorld, tiles: TileLibrary, seed: int) -> List[PickupState]:
    rng = random.Random(seed ^ 0x51A7)
    if isinstance(grid, ChunkedWorld):
        spawn_x = grid.spawn_tile_x
        spawn_y = grid.spawn_tile_y
        width = grid.world_size_tiles
        height = grid.world_size_tiles
    else:
        spawn_x = len(grid[0]) // 2
        spawn_y = len(grid) // 2
        width = len(grid[0])
        height = len(grid)
    candidates: List[Tuple[int, int, int]] = []

    for y in range(2, height - 2):
        for x in range(2, width - 2):
            cell = grid.tile_at(x, y) if isinstance(grid, ChunkedWorld) else int(grid[y][x])
            if int(cell) != 0:
                continue
            if abs(x - spawn_x) + abs(y - spawn_y) < 8:
                continue
            open_neighbors = count_open_neighbors(grid, x, y)
            if open_neighbors < 2:
                continue
            score = open_neighbors * 100 + rng.randint(0, 25)
            candidates.append((score, x, y))

    candidates.sort(reverse=True)
    target_count = max(8, WORLD_SIZE // 8)
    pickups: List[PickupState] = []
    occupied: List[Tuple[int, int]] = []

    for _, x, y in candidates:
        if len(pickups) >= target_count:
            break
        if any(abs(x - ox) <= 2 and abs(y - oy) <= 2 for ox, oy in occupied):
            continue
        sprite_index = stable_index(x, y, seed ^ 0xF00D, max(1, len(tiles.pickup)))
        pickups.append(
            PickupState(
                x=x * TILE_SIZE + TILE_SIZE / 2,
                y=y * TILE_SIZE + TILE_SIZE / 2,
                sprite_index=sprite_index,
                bob_phase=rng.random() * TAU,
                heal=HEALTH_PICKUP,
            )
        )
        occupied.append((x, y))

    return pickups


def collect_pickups(player: PlayerState, pickups: List[PickupState]) -> None:
    for pickup in pickups[:]:
        if (player.x - pickup.x) ** 2 + (player.y - pickup.y) ** 2 <= (PLAYER_RADIUS + 18) ** 2:
            player.health = min(HEALTH_MAX, player.health + pickup.heal)
            pickups.remove(pickup)


def collect_boxes(player: PlayerState, world: ChunkedWorld) -> None:
    for box in world.boxes:
        if box.collected:
            continue
        if (player.x - box.x) ** 2 + (player.y - box.y) ** 2 <= (PLAYER_RADIUS + 20) ** 2:
            box.collected = True
            world.boxes_collected = min(4, world.boxes_collected + 1)


def collect_speed_items(player: PlayerState, world: ChunkedWorld) -> None:
    for item in world.speed_items:
        if item.collected:
            continue
        if (player.x - item.x) ** 2 + (player.y - item.y) ** 2 <= (PLAYER_RADIUS + 20) ** 2:
            item.collected = True
            player.speed_timer = SPEED_ITEM_DURATION


def update_lobby_objective(player: PlayerState, world: ChunkedWorld, submit_pressed: bool) -> None:
    lobby_name = world.area_name_at(world.spawn_tile_x, world.spawn_tile_y)
    current_name = world.area_name_at(int(player.x // TILE_SIZE), int(player.y // TILE_SIZE))
    if submit_pressed and current_name == lobby_name and world.boxes_collected >= 4:
        world.lobby_returned = True


def bullet_spawn(player: PlayerState) -> BulletState:
    angle = player.facing
    return BulletState(
        x=player.x + math.cos(angle) * 18.0,
        y=player.y + math.sin(angle) * 18.0,
        z=player.z + 6.0,
        vx=math.cos(angle) * BULLET_SPEED,
        vy=math.sin(angle) * BULLET_SPEED,
        life=BULLET_LIFETIME,
        angle=angle,
    )


def build_enemies(grid: Sequence[Sequence[int]] | ChunkedWorld, seed: int, target_count: int = ENEMY_COUNT) -> List[EnemyState]:
    rng = random.Random(seed ^ 0xE11F)
    if isinstance(grid, ChunkedWorld):
        spawn_x = grid.spawn_tile_x
        spawn_y = grid.spawn_tile_y
        width = grid.world_size_tiles
        height = grid.world_size_tiles
    else:
        spawn_x = len(grid[0]) // 2
        spawn_y = len(grid) // 2
        width = len(grid[0])
        height = len(grid)
    candidates: List[Tuple[int, int, int]] = []

    for y in range(2, height - 2):
        for x in range(2, width - 2):
            cell = grid.tile_at(x, y) if isinstance(grid, ChunkedWorld) else int(grid[y][x])
            if int(cell) != 0:
                continue
            distance = abs(x - spawn_x) + abs(y - spawn_y)
            if distance < 14:
                continue
            open_neighbors = count_open_neighbors(grid, x, y)
            if open_neighbors < 2:
                continue
            score = distance * 10 + open_neighbors * 120 + rng.randint(0, 20)
            candidates.append((score, x, y))

    candidates.sort(reverse=True)
    enemies: List[EnemyState] = []
    occupied: List[Tuple[int, int]] = []

    for _, x, y in candidates:
        if len(enemies) >= target_count:
            break
        if any(abs(x - ox) <= 3 and abs(y - oy) <= 3 for ox, oy in occupied):
            continue
        enemies.append(
            EnemyState(
                x=x * TILE_SIZE + TILE_SIZE / 2,
                y=y * TILE_SIZE + TILE_SIZE / 2,
                wander_angle=rng.random() * TAU,
                wander_timer=rng.uniform(0.6, 1.8),
                contact_cooldown=rng.uniform(0.0, 0.6),
            )
        )
        occupied.append((x, y))

    return enemies


def resolve_enemy_axis(enemy: EnemyState, grid: Sequence[Sequence[int]] | ChunkedWorld, dx: float, dy: float) -> None:
    if dx != 0.0:
        enemy.x += dx
        rect = rect_from_center(enemy.x, enemy.y, ENEMY_RADIUS * 2, ENEMY_RADIUS * 2)
        if rect_hits_walls(rect, grid):
            enemy.x -= dx
            enemy.vx = 0.0

    if dy != 0.0:
        enemy.y += dy
        rect = rect_from_center(enemy.x, enemy.y, ENEMY_RADIUS * 2, ENEMY_RADIUS * 2)
        if rect_hits_walls(rect, grid):
            enemy.y -= dy
            enemy.vy = 0.0


def update_enemies(enemies: List[EnemyState], player: PlayerState, dt: float, grid: Sequence[Sequence[int]] | ChunkedWorld) -> None:
    for enemy in enemies:
        enemy.contact_cooldown = max(0.0, enemy.contact_cooldown - dt)
        enemy.wander_timer -= dt

        dx = player.x - enemy.x
        dy = player.y - enemy.y
        dist = math.hypot(dx, dy)

        if dist < TILE_SIZE * 6:
            desired_angle = math.atan2(dy, dx)
            enemy.wander_angle = desired_angle
            enemy.wander_timer = 0.8
        elif enemy.wander_timer <= 0.0:
            enemy.wander_angle += random.uniform(-1.1, 1.1)
            enemy.wander_timer = random.uniform(0.6, 1.7)

        if dist > 1e-4:
            steer = 1.0 if dist < TILE_SIZE * 8 else 0.55
            desired_vx = math.cos(enemy.wander_angle) * ENEMY_SPEED * steer
            desired_vy = math.sin(enemy.wander_angle) * ENEMY_SPEED * steer
        else:
            desired_vx = 0.0
            desired_vy = 0.0

        enemy.vx = approach(enemy.vx, desired_vx, 920.0 * dt)
        enemy.vy = approach(enemy.vy, desired_vy, 920.0 * dt)

        move_x = enemy.vx * dt
        move_y = enemy.vy * dt
        resolve_enemy_axis(enemy, grid, move_x, 0.0)
        resolve_enemy_axis(enemy, grid, 0.0, move_y)

        if dist <= PLAYER_RADIUS + ENEMY_RADIUS + 2:
            if enemy.contact_cooldown <= 0.0 and player.hurt_timer <= 0.0:
                player.health = max(0.0, player.health - ENEMY_TOUCH_DAMAGE)
                player.hurt_timer = 0.32
                enemy.contact_cooldown = ENEMY_TOUCH_COOLDOWN
            # small push so they don't stick on top of the player
            enemy.wander_angle += 1.2


def update_bullets(bullets: List[BulletState], enemies: List[EnemyState], dt: float, grid: Sequence[Sequence[int]] | ChunkedWorld) -> None:
    for bullet in bullets[:]:
        bullet.x += bullet.vx * dt
        bullet.y += bullet.vy * dt
        bullet.life -= dt

        rect = rect_from_center(bullet.x, bullet.y, 8, 8)
        if bullet.life <= 0.0 or rect_hits_walls(rect, grid):
            bullets.remove(bullet)
            continue

        hit_enemy = None
        for enemy in enemies:
            if (bullet.x - enemy.x) ** 2 + (bullet.y - enemy.y) ** 2 <= (ENEMY_RADIUS + 5) ** 2:
                hit_enemy = enemy
                break

        if hit_enemy is not None:
            if bullet in bullets:
                bullets.remove(bullet)
            if hit_enemy in enemies:
                enemies.remove(hit_enemy)


def update_player(player: PlayerState, dt: float, input_x: float, input_y: float, grid: Sequence[Sequence[int]] | ChunkedWorld) -> None:
    player.hurt_timer = max(0.0, player.hurt_timer - dt)
    player.speed_timer = max(0.0, player.speed_timer - dt)

    movement_x, movement_y = normalize(input_x, input_y)
    speed_multiplier = SPEED_ITEM_MULTIPLIER if player.speed_timer > 0.0 else 1.0
    desired_vx = movement_x * PLAYER_MAX_SPEED * speed_multiplier
    desired_vy = movement_y * PLAYER_MAX_SPEED * speed_multiplier

    if movement_x != 0.0 or movement_y != 0.0:
        player.vx = approach(player.vx, desired_vx, PLAYER_ACCEL * dt)
        player.vy = approach(player.vy, desired_vy, PLAYER_ACCEL * dt)
    else:
        player.vx = approach(player.vx, 0.0, PLAYER_FRICTION * dt)
        player.vy = approach(player.vy, 0.0, PLAYER_FRICTION * dt)

    move_x = player.vx * dt
    move_y = player.vy * dt
    resolve_player_axis(player, grid, move_x, 0.0)
    resolve_player_axis(player, grid, 0.0, move_y)

    if isinstance(grid, ChunkedWorld):
        max_x = grid.world_size_tiles * TILE_SIZE - PLAYER_RADIUS
        max_y = grid.world_size_tiles * TILE_SIZE - PLAYER_RADIUS
    else:
        max_x = WORLD_SIZE * TILE_SIZE - PLAYER_RADIUS
        max_y = WORLD_SIZE * TILE_SIZE - PLAYER_RADIUS
    player.x = clamp(player.x, PLAYER_RADIUS, max_x)
    player.y = clamp(player.y, PLAYER_RADIUS, max_y)

    player.z = 0.0
    player.vz = 0.0
    player.grounded = True


# ---------------------------
# Tile loading / rendering
# ---------------------------
def build_fallback_floor(color_a: Tuple[int, int, int], color_b: Tuple[int, int, int]) -> pygame.Surface:
    surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
    surf.fill(color_a)
    pygame.draw.rect(surf, color_b, (4, 4, TILE_SIZE - 8, TILE_SIZE - 8), 2, border_radius=6)
    pygame.draw.line(surf, (255, 255, 255, 40), (0, TILE_SIZE // 3), (TILE_SIZE, TILE_SIZE // 2), 2)
    pygame.draw.line(surf, (0, 0, 0, 30), (0, TILE_SIZE * 2 // 3), (TILE_SIZE, TILE_SIZE // 2), 2)
    return surf


def build_fallback_wall() -> pygame.Surface:
    surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
    surf.fill((92, 72, 58))
    pygame.draw.rect(surf, (132, 104, 82), (4, 4, TILE_SIZE - 8, TILE_SIZE - 8), 2, border_radius=6)
    for y in (14, 28, 42):
        pygame.draw.line(surf, (64, 50, 42), (10, y), (TILE_SIZE - 10, y), 2)
    return surf


def build_fallback_pickup() -> pygame.Surface:
    surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
    pygame.draw.circle(surf, (255, 65, 65), (TILE_SIZE // 2, TILE_SIZE // 2), 12)
    pygame.draw.rect(surf, (255, 255, 255), (TILE_SIZE // 2 - 3, 14, 6, 36), border_radius=2)
    pygame.draw.rect(surf, (255, 255, 255), (14, TILE_SIZE // 2 - 3, 36, 6), border_radius=2)
    return surf


def build_fallback_box() -> pygame.Surface:
    surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
    pygame.draw.rect(surf, (155, 112, 66), (12, 14, 40, 36), border_radius=4)
    pygame.draw.rect(surf, (210, 166, 104), (12, 14, 40, 36), 3, border_radius=4)
    pygame.draw.line(surf, (120, 84, 48), (12, 32), (52, 32), 2)
    pygame.draw.line(surf, (120, 84, 48), (32, 14), (32, 50), 2)
    return surf


def load_named_surfaces(tile_dir: Path, names: Sequence[str]) -> List[pygame.Surface]:
    surfaces: List[pygame.Surface] = []
    for name in names:
        path = tile_dir / name
        if not path.exists():
            continue
        try:
            surface = pygame.image.load(str(path)).convert_alpha()
            if surface.get_size() != (TILE_SIZE, TILE_SIZE):
                surface = pygame.transform.smoothscale(surface, (TILE_SIZE, TILE_SIZE))
            surfaces.append(surface)
        except Exception:
            continue
    return surfaces


def load_named_surface(tile_dir: Path, name: str) -> pygame.Surface | None:
    path = tile_dir / name
    if not path.exists():
        return None
    try:
        surface = pygame.image.load(str(path)).convert_alpha()
        if surface.get_size() != (TILE_SIZE, TILE_SIZE):
            surface = pygame.transform.smoothscale(surface, (TILE_SIZE, TILE_SIZE))
        return surface
    except Exception:
        return None


def load_surfaces_from_dir(tile_dir: Path) -> List[pygame.Surface]:
    surfaces: List[pygame.Surface] = []
    if not tile_dir.exists():
        return surfaces

    for path in sorted(tile_dir.glob("*.png"), key=natural_key):
        try:
            surface = pygame.image.load(str(path)).convert_alpha()
            if surface.get_size() != (TILE_SIZE, TILE_SIZE):
                surface = pygame.transform.smoothscale(surface, (TILE_SIZE, TILE_SIZE))
            surfaces.append(surface)
        except Exception:
            continue
    return surfaces


def recolor_surface(surface: pygame.Surface, tint: Tuple[int, int, int], alpha: int) -> pygame.Surface:
    tinted = surface.copy()
    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    overlay.fill((*tint, alpha))
    tinted.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
    return tinted


def build_texture_variants(
    surfaces: Sequence[pygame.Surface],
    base_tint: Tuple[int, int, int],
    accent_tint: Tuple[int, int, int],
) -> List[pygame.Surface]:
    variants: List[pygame.Surface] = []
    for i, surface in enumerate(surfaces):
        variants.append(surface)
        variants.append(pygame.transform.flip(surface, True, False))
        if i % 2 == 0:
            variants.append(shade_surface(surface, light=8, dark=0))
        else:
            variants.append(shade_surface(surface, light=0, dark=10))
    unique: List[pygame.Surface] = []
    seen: set[int] = set()
    for surf in variants:
        key = id(surf)
        if key in seen:
            continue
        seen.add(key)
        unique.append(surf)
    return unique


def sort_surfaces_for_category(surfaces: Sequence[pygame.Surface], category: str) -> List[pygame.Surface]:
    scored: List[Tuple[float, pygame.Surface]] = []
    for surface in surfaces:
        (_, _, _), transparency, brightness = surface_rgb_and_transparency(surface)
        avg = pygame.transform.average_color(surface)
        if hasattr(avg, "r"):
            r, g, b = int(avg.r), int(avg.g), int(avg.b)
        else:
            r, g, b = int(avg[0]), int(avg[1]), int(avg[2])
        saturation = max(r, g, b) - min(r, g, b)
        if category == "floor":
            score = -abs(brightness - 150.0) - saturation * 0.6 - transparency * 40.0
        elif category == "route":
            score = -abs(brightness - 120.0) - saturation * 0.4 - transparency * 30.0
        elif category == "wall":
            score = -abs(brightness - 95.0) - saturation * 0.3 - transparency * 20.0
        else:
            score = brightness
        scored.append((score, surface))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [surface for _, surface in scored]


def shade_surface(surface: pygame.Surface, light: int = 0, dark: int = 0) -> pygame.Surface:
    shaded = surface.copy()
    if light:
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        overlay.fill((255, 255, 255, light))
        shaded.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
    if dark:
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, dark))
        shaded.blit(overlay, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return shaded


def surface_rgb_and_transparency(surface: pygame.Surface) -> Tuple[Tuple[int, int, int], float, float]:
    avg = pygame.transform.average_color(surface)
    if hasattr(avg, "r"):
        r, g, b = int(avg.r), int(avg.g), int(avg.b)
    else:
        r, g, b = int(avg[0]), int(avg[1]), int(avg[2])
    bbox = surface.get_bounding_rect()
    area = float(surface.get_width() * surface.get_height())
    bbox_ratio = 0.0 if area == 0 else (bbox.width * bbox.height) / area
    transparency = 1.0 - bbox_ratio
    brightness = (r + g + b) / 3.0
    return (r, g, b), transparency, brightness


def classify_tile_surface(surface: pygame.Surface) -> str:
    rgb, transparency, brightness = surface_rgb_and_transparency(surface)
    r, g, b = rgb
    if transparency > 0.08:
        return "pickup"
    if g > r + 15 or b > r + 15 or (r > 150 and g > 120 and b > 80) or brightness > 170:
        return "floor"
    return "wall"


def load_tile_library() -> TileLibrary:
    uses_dir = Path("uses")
    floor = load_named_surfaces(uses_dir / "Floor", [
        "wood 1.png",
        "wood 2.png",
        "wood 3.png",
        "wood 4.png",
        "wood 5.png",
        "wood 6.png",
    ])
    route = load_surfaces_from_dir(uses_dir / "Route")
    rooms = load_surfaces_from_dir(uses_dir / "Rooms")
    box = load_surfaces_from_dir(uses_dir / "Box")
    wall = load_named_surfaces(uses_dir / "Wall", [
        "first end  wall block left.png",
        "left middle wall block no center.png",
        "fill-Center color block.png",
        "right middle wall block no center.png",
        "second end wall block right.png",
        "Top-middle wall blocks.png",
        "Bottom-middle wall blocks.png",
        "3rd end wall block bottom left.png",
        "4th end wall block bottom right.png",
    ])
    pickup = load_surfaces_from_dir(uses_dir / "Pickup")
    wall_by_role = {
        "center": load_named_surface(uses_dir / "Wall", "fill-Center color block.png"),
        "left": load_named_surface(uses_dir / "Wall", "left middle wall block no center.png"),
        "right": load_named_surface(uses_dir / "Wall", "right middle wall block no center.png"),
        "top": load_named_surface(uses_dir / "Wall", "Top-middle wall blocks.png"),
        "bottom": load_named_surface(uses_dir / "Wall", "Bottom-middle wall blocks.png"),
        "top_left": load_named_surface(uses_dir / "Wall", "first end  wall block left.png"),
        "top_right": None,
        "bottom_left": load_named_surface(uses_dir / "Wall", "3rd end wall block bottom left.png"),
        "bottom_right": load_named_surface(uses_dir / "Wall", "4th end wall block bottom right.png"),
    }

    if wall_by_role["top_left"] is not None and wall_by_role["top_right"] is None:
        wall_by_role["top_right"] = pygame.transform.flip(wall_by_role["top_left"], True, False)
    if wall_by_role["bottom_left"] is not None and wall_by_role["top_left"] is None:
        wall_by_role["top_left"] = pygame.transform.flip(wall_by_role["bottom_left"], False, True)
    if wall_by_role["bottom_right"] is not None and wall_by_role["top_right"] is None:
        wall_by_role["top_right"] = pygame.transform.flip(wall_by_role["bottom_right"], False, True)

    floor_fallback = build_fallback_floor((170, 196, 198), (136, 168, 174))
    route_fallback = build_fallback_floor((150, 182, 190), (116, 146, 156))
    box_fallback = build_fallback_box()
    wall_fallback = build_fallback_wall()
    pickup_fallback = build_fallback_pickup()

    if not floor:
        floor = [floor_fallback]
    if not route:
        route = [route_fallback]
    if not rooms:
        rooms = floor[:]
    if not box:
        box = [box_fallback]
    if not wall:
        wall = [wall_fallback]
    if not pickup:
        pickup = [pickup_fallback]

    return TileLibrary(
        floor=floor,
        route=route,
        rooms=rooms,
        box=box,
        wall=wall,
        wall_by_role=wall_by_role,
        pickup=pickup,
        floor_fallback=floor_fallback,
        route_fallback=route_fallback,
        box_fallback=box_fallback,
        wall_fallback=wall_fallback,
        pickup_fallback=pickup_fallback,
    )


def build_background_surface() -> pygame.Surface:
    surf = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
    top = (10, 12, 22)
    bottom = (26, 36, 52)
    for y in range(VIEW_HEIGHT):
        t = y / max(1, VIEW_HEIGHT - 1)
        color = (
            int(top[0] * (1 - t) + bottom[0] * t),
            int(top[1] * (1 - t) + bottom[1] * t),
            int(top[2] * (1 - t) + bottom[2] * t),
        )
        pygame.draw.line(surf, color, (0, y), (VIEW_WIDTH, y))

    pygame.draw.circle(surf, (226, 196, 136), (VIEW_WIDTH - 170, 110), 40)
    glow = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT), pygame.SRCALPHA)
    pygame.draw.circle(glow, (255, 190, 100, 60), (VIEW_WIDTH - 170, 110), 140)
    pygame.draw.circle(glow, (255, 190, 100, 24), (VIEW_WIDTH - 170, 110), 270)
    pygame.draw.ellipse(glow, (70, 110, 170, 30), (VIEW_WIDTH * 0.05, VIEW_HEIGHT * 0.18, 610, 220))
    pygame.draw.ellipse(glow, (20, 30, 50, 40), (VIEW_WIDTH * 0.58, VIEW_HEIGHT * 0.04, 520, 300))
    surf.blit(glow, (0, 0))
    return surf


def pick_variant(variants: Sequence[pygame.Surface], x: int, y: int, salt: int, fallback: pygame.Surface) -> pygame.Surface:
    if not variants:
        return fallback
    return variants[stable_index(x, y, salt, len(variants))]


def select_floor_surface(tiles: TileLibrary, grid: Sequence[Sequence[int]] | ChunkedWorld, x: int, y: int, seed: int) -> pygame.Surface:
    if isinstance(grid, ChunkedWorld):
        chunk_x, chunk_y = grid.tile_to_chunk(x, y)
        role = chunk_role(chunk_x, chunk_y, grid)
        if role == "route" and tiles.route:
            return tiles.route[stable_index(chunk_x, chunk_y, 0x7075, len(tiles.route))]
        salt = 0xA100 if role in {"room", "hub"} else 0xF100
        return tiles.floor[stable_index(chunk_x, chunk_y, salt, len(tiles.floor))] if tiles.floor else tiles.floor_fallback
    return tiles.floor[stable_index(x, y, 0xF100, len(tiles.floor))] if tiles.floor else tiles.floor_fallback


def select_route_surface(tiles: TileLibrary, grid: Sequence[Sequence[int]] | ChunkedWorld, x: int, y: int, seed: int) -> pygame.Surface:
    if isinstance(grid, ChunkedWorld):
        chunk_x, chunk_y = grid.tile_to_chunk(x, y)
        role = chunk_role(chunk_x, chunk_y, grid)
        if role == "route" and tiles.route:
            return tiles.route[stable_index(chunk_x, chunk_y, 0x7075, len(tiles.route))]
        if tiles.floor:
            return tiles.floor[stable_index(chunk_x, chunk_y, 0xF100, len(tiles.floor))]
        return tiles.floor_fallback
    if tiles.route:
        return tiles.route[stable_index(x, y, 0x7075, len(tiles.route))]
    return tiles.route_fallback


def select_wall_surface(tiles: TileLibrary, grid: Sequence[Sequence[int]] | ChunkedWorld, x: int, y: int, seed: int) -> pygame.Surface:
    def cell_at(nx: int, ny: int) -> int:
        if isinstance(grid, ChunkedWorld):
            return grid.tile_at(nx, ny)
        if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]):
            return int(grid[ny][nx])
        return 1

    up = cell_at(x, y - 1)
    down = cell_at(x, y + 1)
    left = cell_at(x - 1, y)
    right = cell_at(x + 1, y)
    openings = {
        "up": up == 0,
        "down": down == 0,
        "left": left == 0,
        "right": right == 0,
    }
    open_count = sum(1 for value in openings.values() if value)
    if open_count == 1:
        if openings["up"]:
            role = "top"
        elif openings["down"]:
            role = "bottom"
        elif openings["left"]:
            role = "left"
        else:
            role = "right"
    elif open_count == 2:
        if openings["up"] and openings["left"]:
            role = "top_left"
        elif openings["up"] and openings["right"]:
            role = "top_right"
        elif openings["down"] and openings["left"]:
            role = "bottom_left"
        elif openings["down"] and openings["right"]:
            role = "bottom_right"
        elif openings["left"] and openings["right"]:
            role = "center"
        else:
            role = "center"
    elif open_count == 3:
        if not openings["up"]:
            role = "top"
        elif not openings["down"]:
            role = "bottom"
        elif not openings["left"]:
            role = "left"
        else:
            role = "right"
    else:
        role = "center"
    surface = tiles.wall_by_role.get(role)
    if surface is not None:
        return surface
    if tiles.wall:
        return tiles.wall[0]
    return tiles.wall_fallback


def draw_background(surface: pygame.Surface, background: pygame.Surface, camera: Camera) -> None:
    surface.blit(background, (0, 0))
    haze = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT), pygame.SRCALPHA)
    offset_x = int(camera.x * 0.015)
    offset_y = int(camera.y * 0.010)
    pygame.draw.ellipse(haze, (90, 130, 180, 24), (80 + offset_x, 120 + offset_y, 540, 180))
    pygame.draw.ellipse(haze, (20, 28, 44, 26), (VIEW_WIDTH - 610 - offset_x, 70 - offset_y, 520, 250))
    pygame.draw.ellipse(haze, (255, 230, 170, 18), (VIEW_WIDTH - 220 + offset_x, 64 + offset_y, 120, 120))
    surface.blit(haze, (0, 0))


def draw_shadow(surface: pygame.Surface, camera: Camera, x: float, y: float, z: float, strength: float) -> None:
    sx, sy = world_to_screen(camera, x, y)
    shadow = pygame.Surface((36, 20), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow, (0, 0, 0, int(90 * strength)), shadow.get_rect())
    shadow = pygame.transform.smoothscale(
        shadow,
        (
            int(36 * (1.0 + strength * 0.4)),
            int(20 * (1.0 + strength * 0.2)),
        ),
    )
    surface.blit(shadow, shadow.get_rect(center=(sx, sy + 10)))


def draw_player(
    surface: pygame.Surface,
    camera: Camera,
    player: PlayerState,
    elapsed: float,
    player_image: pygame.Surface | None = None,
) -> None:
    sx, sy = world_to_screen(camera, player.x, player.y, player.z)
    draw_shadow(surface, camera, player.x, player.y, player.z, 1.0)

    size = PLAYER_RADIUS * 2 + 6
    rect = pygame.Rect(0, 0, size, size)
    rect.center = (sx, sy)
    pygame.draw.rect(surface, (70, 10, 10), rect.inflate(4, 4), border_radius=3)
    pygame.draw.rect(surface, player.color, rect, border_radius=3)

    if player_image is not None:
        image = pygame.transform.smoothscale(player_image, (size - 4, size - 4))
        surface.blit(image, image.get_rect(center=rect.center))
    else:
        eye_y = rect.top + 7
        pygame.draw.circle(surface, (245, 248, 252), (rect.left + 7, eye_y), 3)
        pygame.draw.circle(surface, (245, 248, 252), (rect.right - 7, eye_y), 3)
        pygame.draw.circle(surface, (20, 24, 32), (rect.left + 7, eye_y), 1)
        pygame.draw.circle(surface, (20, 24, 32), (rect.right - 7, eye_y), 1)

    if player.hurt_timer > 0.0:
        pulse = 0.5 + 0.5 * math.sin(elapsed * 20.0)
        flash_color = (255, 255, 255) if pulse > 0.5 else (255, 150, 150)
        pygame.draw.rect(surface, flash_color, rect.inflate(8, 8), 2, border_radius=4)


def draw_bullet(surface: pygame.Surface, camera: Camera, bullet: BulletState) -> None:
    sx, sy = world_to_screen(camera, bullet.x, bullet.y, bullet.z)
    projectile = pygame.Surface((12, 12), pygame.SRCALPHA)
    pygame.draw.circle(projectile, (255, 226, 110), (6, 6), 4)
    pygame.draw.circle(projectile, (255, 255, 240), (4, 4), 1)
    surface.blit(projectile, projectile.get_rect(center=(sx, sy)))


def draw_pickup(surface: pygame.Surface, camera: Camera, pickup: PickupState, tiles: TileLibrary, elapsed: float) -> None:
    sx, sy = world_to_screen(camera, pickup.x, pickup.y, 0.0)
    bob = math.sin(elapsed * 4.5 + pickup.bob_phase) * 4.0

    glow = pygame.Surface((40, 40), pygame.SRCALPHA)
    pygame.draw.circle(glow, (80, 220, 120, 96), (20, 20), 16)
    pygame.draw.circle(glow, (255, 255, 255, 32), (20, 20), 9)
    surface.blit(glow, glow.get_rect(center=(sx, int(sy - bob))))

    if tiles.pickup:
        sprite = tiles.pickup[pickup.sprite_index % len(tiles.pickup)]
    else:
        sprite = tiles.pickup_fallback
    sprite = pygame.transform.rotozoom(sprite, math.sin(elapsed * 2.0 + pickup.bob_phase) * 6.0, 0.56)
    surface.blit(sprite, sprite.get_rect(center=(sx, int(sy - bob))))


def draw_box(surface: pygame.Surface, camera: Camera, box: BoxState, tiles: TileLibrary, elapsed: float) -> None:
    if box.collected:
        return
    sx, sy = world_to_screen(camera, box.x, box.y, 0.0)
    bob = math.sin(elapsed * 3.8 + box.bob_phase) * 4.0

    glow = pygame.Surface((46, 46), pygame.SRCALPHA)
    pygame.draw.circle(glow, (55, 140, 255, 90), (23, 23), 17)
    pygame.draw.circle(glow, (255, 255, 255, 28), (23, 23), 9)
    surface.blit(glow, glow.get_rect(center=(sx, int(sy - bob))))

    sprite = tiles.box[box.sprite_index % len(tiles.box)] if tiles.box else tiles.box_fallback
    sprite = pygame.transform.rotozoom(sprite, math.sin(elapsed * 1.6 + box.bob_phase) * 5.0, 0.72)
    surface.blit(sprite, sprite.get_rect(center=(sx, int(sy - bob))))


def draw_speed_item(surface: pygame.Surface, camera: Camera, item: SpeedItemState, elapsed: float) -> None:
    if item.collected:
        return
    sx, sy = world_to_screen(camera, item.x, item.y, 0.0)
    bob = math.sin(elapsed * 5.2 + item.bob_phase) * 4.0
    center = (sx, int(sy - bob))

    glow = pygame.Surface((52, 52), pygame.SRCALPHA)
    pygame.draw.circle(glow, (60, 170, 255, 110), (26, 26), 20)
    pygame.draw.circle(glow, (160, 235, 255, 55), (26, 26), 12)
    surface.blit(glow, glow.get_rect(center=center))

    points = [
        (center[0] - 4, center[1] - 15),
        (center[0] + 9, center[1] - 15),
        (center[0] + 2, center[1] - 2),
        (center[0] + 12, center[1] - 2),
        (center[0] - 6, center[1] + 17),
        (center[0] - 1, center[1] + 3),
        (center[0] - 12, center[1] + 3),
    ]
    pygame.draw.polygon(surface, (245, 248, 252), points)
    pygame.draw.polygon(surface, (55, 150, 255), points, 2)


def draw_enemy(surface: pygame.Surface, camera: Camera, enemy: EnemyState, elapsed: float) -> None:
    sx, sy = world_to_screen(camera, enemy.x, enemy.y, 0.0)
    shadow = pygame.Surface((34, 18), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow, (0, 0, 0, 90), shadow.get_rect())
    surface.blit(shadow, shadow.get_rect(center=(sx, sy + 8)))

    bob = math.sin(elapsed * 6.0 + (enemy.x + enemy.y) * 0.01) * 1.2
    center = (sx, int(sy + bob))
    pygame.draw.circle(surface, (35, 10, 10), center, ENEMY_RADIUS + 3)
    pygame.draw.circle(surface, (170, 42, 42), center, ENEMY_RADIUS)
    pygame.draw.circle(surface, (255, 210, 90), (center[0] - 3, center[1] - 2), 2)
    pygame.draw.circle(surface, (255, 210, 90), (center[0] + 3, center[1] - 2), 2)


def draw_hud(
    surface: pygame.Surface,
    player: PlayerState,
    world: ChunkedWorld,
    credit_font: pygame.font.Font,
    text_font: pygame.font.Font,
) -> None:
    credit = credit_font.render("Developed/Designed by Yahia & Nizar", True, (240, 244, 250))
    surface.blit(credit, (16, 16))

    bar_x = 16
    bar_y = 44
    bar_w = 220
    bar_h = 16
    health_ratio = clamp(player.health / HEALTH_MAX, 0.0, 1.0)
    if health_ratio > 0.66:
        fill_color = (72, 210, 108)
    elif health_ratio > 0.33:
        fill_color = (235, 183, 56)
    else:
        fill_color = (220, 64, 64)

    pygame.draw.rect(surface, (8, 10, 16, 160), (bar_x, bar_y, bar_w, bar_h), border_radius=8)
    pygame.draw.rect(surface, fill_color, (bar_x, bar_y, int(bar_w * health_ratio), bar_h), border_radius=8)
    pygame.draw.rect(surface, (240, 244, 250), (bar_x, bar_y, bar_w, bar_h), 1, border_radius=8)

    health_text = text_font.render(f"HEALTH {int(player.health)}/{int(HEALTH_MAX)}", True, (240, 244, 250))
    surface.blit(health_text, (bar_x, 64))

    area_name = world.area_name_at(int(player.x // TILE_SIZE), int(player.y // TILE_SIZE))
    area_text = text_font.render(f"AREA {area_name}", True, (240, 244, 250))
    surface.blit(area_text, (16, 90))

    box_text = text_font.render(f"BOXES {world.boxes_collected}/4", True, (240, 244, 250))
    surface.blit(box_text, (16, 114))

    objective = "Return to Lobby" if world.boxes_collected < 4 else "Return to Lobby to finish"
    objective_text = text_font.render(objective, True, (180, 214, 255))
    surface.blit(objective_text, (16, 138))

    time_text = text_font.render(f"TIME {format_time(world.time_left)}", True, (245, 248, 252))
    surface.blit(time_text, (16, 162))

    speed_text = text_font.render(
        f"SPEED x{SPEED_ITEM_MULTIPLIER:.0f} {max(0, int(math.ceil(player.speed_timer)))}s" if player.speed_timer > 0 else "SPEED x1",
        True,
        (180, 240, 200) if player.speed_timer > 0 else (245, 248, 252),
    )
    surface.blit(speed_text, (16, 186))

    if world.lobby_returned:
        success_text = text_font.render("Quest complete", True, (80, 190, 255))
        surface.blit(success_text, (16, 210))


def draw_world(
    surface: pygame.Surface,
    world: ChunkedWorld,
    camera: Camera,
    seed: int,
    player: PlayerState,
    elapsed: float,
    player_image: pygame.Surface | None = None,
) -> None:
    left = max(0, int((camera.x - VIEW_WIDTH / 2) // TILE_SIZE) - 1)
    right = min(world.world_size_tiles - 1, int((camera.x + VIEW_WIDTH / 2) // TILE_SIZE) + 2)
    top = max(0, int((camera.y - VIEW_HEIGHT / 2) // TILE_SIZE) - 1)
    bottom = min(world.world_size_tiles - 1, int((camera.y + VIEW_HEIGHT / 2) // TILE_SIZE) + 2)

    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            sx, sy = world_to_screen(camera, x * TILE_SIZE, y * TILE_SIZE)
            cell = world.tile_at(x, y)
            open_neighbors = count_open_neighbors(world, x, y)
            if cell == 2:
                floor_tile = select_route_surface(world.tiles, world, x, y, seed)
            else:
                floor_tile = select_floor_surface(world.tiles, world, x, y, seed)
            surface.blit(floor_tile, (sx, sy))
            if cell == 1:
                wall_tile = select_wall_surface(world.tiles, world, x, y, seed)
                surface.blit(wall_tile, (sx, sy))
            elif open_neighbors <= 1:
                edge = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(edge, (255, 255, 255, 10), (0, 0, TILE_SIZE, TILE_SIZE), 1)
                pygame.draw.line(edge, (255, 255, 255, 18), (0, 0), (TILE_SIZE - 1, 0), 1)
                pygame.draw.line(edge, (0, 0, 0, 18), (0, TILE_SIZE - 1), (TILE_SIZE - 1, TILE_SIZE - 1), 1)
                surface.blit(edge, (sx, sy))
            elif open_neighbors >= 3:
                edge = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                pygame.draw.rect(edge, (0, 0, 0, 8), (0, 0, TILE_SIZE, TILE_SIZE), 1)
                surface.blit(edge, (sx, sy))

    for pickup in world.pickups:
        draw_pickup(surface, camera, pickup, world.tiles, elapsed)

    for box in world.boxes:
        draw_box(surface, camera, box, world.tiles, elapsed)

    for item in world.speed_items:
        draw_speed_item(surface, camera, item, elapsed)

    for enemy in world.enemies:
        draw_enemy(surface, camera, enemy, elapsed)

    for bullet in getattr(world, "bullets", []):
        draw_bullet(surface, camera, bullet)

    lobby_world_x = world.spawn_world_x
    lobby_world_y = world.spawn_world_y
    lobby_sx, lobby_sy = world_to_screen(camera, lobby_world_x, lobby_world_y, 0.0)
    lobby_label_font = pygame.font.Font(None, 24)
    if -64 <= lobby_sx <= VIEW_WIDTH + 64 and -64 <= lobby_sy <= VIEW_HEIGHT + 64:
        pygame.draw.circle(surface, (60, 150, 255), (lobby_sx, lobby_sy), 12)
        pygame.draw.circle(surface, (220, 240, 255), (lobby_sx, lobby_sy), 12, 2)
        lobby_text = lobby_label_font.render("Lobby", True, (235, 244, 252))
        surface.blit(lobby_text, lobby_text.get_rect(center=(lobby_sx, lobby_sy + 26)))

    label_font = pygame.font.Font(None, 20)
    for (chunk_x, chunk_y), name in world.area_names.items():
        tile_x = chunk_x * world.chunk_size + world.chunk_size // 2
        tile_y = chunk_y * world.chunk_size + world.chunk_size // 2
        sx, sy = world_to_screen(camera, tile_x * TILE_SIZE, tile_y * TILE_SIZE, 0.0)
        if -80 <= sx <= VIEW_WIDTH + 80 and -80 <= sy <= VIEW_HEIGHT + 80:
            label = label_font.render(name, True, (240, 244, 250))
            shadow = label_font.render(name, True, (20, 24, 32))
            surface.blit(shadow, shadow.get_rect(center=(sx + 1, sy + 1)))
            surface.blit(label, label.get_rect(center=(sx, sy)))

    draw_player(surface, camera, player, elapsed, player_image)


def draw_loading_screen(
    surface: pygame.Surface,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    progress: float,
    headline: str,
    detail: str,
) -> None:
    progress = clamp(progress, 0.0, 1.0)
    surface.fill((0, 0, 0))

    title = title_font.render(headline, True, (245, 248, 252))
    detail_text = body_font.render(detail, True, (200, 210, 225))
    percent_text = body_font.render(f"{int(progress * 100)}%", True, (245, 248, 252))
    surface.blit(title, title.get_rect(center=(VIEW_WIDTH // 2, VIEW_HEIGHT // 2 - 54)))
    surface.blit(detail_text, detail_text.get_rect(center=(VIEW_WIDTH // 2, VIEW_HEIGHT // 2 - 18)))
    surface.blit(percent_text, percent_text.get_rect(center=(VIEW_WIDTH // 2, VIEW_HEIGHT // 2 + 52)))

    bar_w = 540
    bar_h = 18
    bar_x = (VIEW_WIDTH - bar_w) // 2
    bar_y = VIEW_HEIGHT // 2 + 12
    pygame.draw.rect(surface, (16, 20, 30), (bar_x, bar_y, bar_w, bar_h), border_radius=10)
    pygame.draw.rect(surface, (92, 182, 255), (bar_x, bar_y, int(bar_w * progress), bar_h), border_radius=10)
    pygame.draw.rect(surface, (245, 248, 252), (bar_x, bar_y, bar_w, bar_h), 1, border_radius=10)

    pygame.display.flip()
    pygame.event.pump()


def draw_menu_screen(
    surface: pygame.Surface,
    fonts: dict,
    profile: dict,
    background_image: pygame.Surface | None,
    title_image: pygame.Surface | None,
    play_image: pygame.Surface | None,
    customization_image: pygame.Surface | None,
    exit_image: pygame.Surface | None,
    mouse_pos: Tuple[int, int],
    t: float,
) -> dict:
    if background_image is not None:
        surface.blit(scale_cover(background_image, (VIEW_WIDTH, VIEW_HEIGHT)), (0, 0))
    else:
        draw_background(surface, build_background_surface(), Camera(0.0, 0.0))

    veil = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT), pygame.SRCALPHA)
    veil.fill((0, 0, 0, 120))
    surface.blit(veil, (0, 0))

    title_font = fonts["title"]
    body_font = fonts["body"]
    button_font = fonts["button"]

    title_rect = pygame.Rect(0, 0, 560, 150)
    title_rect.centerx = VIEW_WIDTH // 2
    title_rect.y = 36
    if title_image is not None:
        max_w = 1280
        max_h = 360
        src_w, src_h = title_image.get_size()
        scale = min(max_w / max(1, src_w), max_h / max(1, src_h))
        draw_w = max(1, int(src_w * scale))
        draw_h = max(1, int(src_h * scale))
        img = title_image if (draw_w, draw_h) == title_image.get_size() else pygame.transform.smoothscale(title_image, (draw_w, draw_h))
        img_rect = img.get_rect()
        img_rect.midtop = (VIEW_WIDTH // 2, -16)
        surface.blit(img, img_rect)
    else:
        pygame.draw.rect(surface, (22, 30, 42), title_rect, border_radius=6)
        pygame.draw.rect(surface, (90, 140, 220), title_rect, 2, border_radius=6)
        draw_centered(surface, title_font, SCREEN_TITLE, (245, 248, 252), title_rect.center)

    draw_centered(surface, body_font, f"Welcome, {profile['name']}", (240, 244, 250), (VIEW_WIDTH // 2, 220))

    pulse = 0.5 + 0.5 * math.sin(t * 2.5)
    accent = (int(70 + pulse * 50), int(160 + pulse * 40), 255)

    buttons = {
        "start": pygame.Rect(0, 0, 320, 78),
        "customization": pygame.Rect(0, 0, 320, 78),
        "exit": pygame.Rect(0, 0, 320, 78),
    }
    buttons["start"].center = (VIEW_WIDTH // 2, 320)
    buttons["customization"].center = (VIEW_WIDTH // 2, 402)
    buttons["exit"].center = (VIEW_WIDTH // 2, 484)

    draw_image_button(surface, play_image, buttons["start"], mouse_pos, "Play", button_font, accent)
    draw_image_button(surface, customization_image, buttons["customization"], mouse_pos, "Customization", button_font, accent)
    draw_image_button(surface, exit_image, buttons["exit"], mouse_pos, "Exit", button_font, accent)

    hint = "Collect 4 boxes, return to Lobby, press T."
    draw_centered(surface, body_font, hint, (210, 220, 235), (VIEW_WIDTH // 2, VIEW_HEIGHT - 40))

    return buttons


def draw_name_entry_screen(
    surface: pygame.Surface,
    fonts: dict,
    buffer: str,
    mouse_pos: Tuple[int, int],
) -> dict:
    surface.fill((0, 0, 0))
    title_font = fonts["title"]
    body_font = fonts["body"]
    button_font = fonts["button"]

    draw_centered(surface, title_font, "Enter Your Name", (245, 248, 252), (VIEW_WIDTH // 2, 180))
    draw_centered(surface, body_font, "This is saved once and reused later.", (180, 190, 205), (VIEW_WIDTH // 2, 232))

    box = pygame.Rect(VIEW_WIDTH // 2 - 220, 292, 440, 54)
    pygame.draw.rect(surface, (24, 30, 40), box, border_radius=8)
    pygame.draw.rect(surface, (90, 140, 220), box, 2, border_radius=8)
    draw_text(surface, button_font, buffer or "Type here", (245, 248, 252), (box.x + 16, box.y + 14))

    ok = pygame.Rect(VIEW_WIDTH // 2 - 110, 376, 220, 46)
    draw_button(surface, ok, "Continue", button_font, mouse_pos, (90, 140, 255))
    return {"continue": ok}


def draw_customization_screen(
    surface: pygame.Surface,
    fonts: dict,
    profile: dict,
    player_image: pygame.Surface | None,
    mouse_pos: Tuple[int, int],
    t: float,
) -> dict:
    surface.fill((10, 12, 18))
    title_font = fonts["title"]
    body_font = fonts["body"]
    button_font = fonts["button"]

    draw_centered(surface, title_font, "Customization", (245, 248, 252), (VIEW_WIDTH // 2, 58))
    draw_text(surface, body_font, "Left/Right: color", (210, 220, 235), (82, 112))
    draw_text(surface, body_font, "I: toggle image", (210, 220, 235), (82, 146))
    draw_text(surface, body_font, "Enter: save", (210, 220, 235), (82, 180))
    draw_text(surface, body_font, "Esc: back", (210, 220, 235), (82, 214))

    preview = pygame.Rect(0, 0, 240, 240)
    preview.center = (VIEW_WIDTH // 2, VIEW_HEIGHT // 2 - 10)
    pygame.draw.circle(surface, (18, 22, 30), preview.center, 118)
    pygame.draw.circle(surface, PLAYER_COLORS[profile["color_index"]], preview.center, 96)
    pygame.draw.circle(surface, (245, 248, 252), preview.center, 96, 2)

    size = 96
    if profile.get("use_player_image", False) and player_image is not None:
        scaled = pygame.transform.smoothscale(player_image, (size, size))
        surface.blit(scaled, scaled.get_rect(center=preview.center))
    else:
        inner = pygame.Rect(0, 0, size, size)
        inner.center = preview.center
        pygame.draw.rect(surface, PLAYER_COLORS[profile["color_index"]], inner, border_radius=4)
        pygame.draw.circle(surface, (245, 248, 252), (inner.left + 22, inner.top + 32), 6)
        pygame.draw.circle(surface, (245, 248, 252), (inner.right - 22, inner.top + 32), 6)
        pygame.draw.circle(surface, (20, 24, 32), (inner.left + 22, inner.top + 32), 2)
        pygame.draw.circle(surface, (20, 24, 32), (inner.right - 22, inner.top + 32), 2)

    draw_centered(surface, body_font, f"Color {profile['color_index'] + 1}/{len(PLAYER_COLORS)}", (235, 240, 250), (VIEW_WIDTH // 2, VIEW_HEIGHT // 2 + 130))
    draw_centered(surface, body_font, f"Image: {'On' if profile.get('use_player_image', False) else 'Off'}", (235, 240, 250), (VIEW_WIDTH // 2, VIEW_HEIGHT // 2 + 160))

    back = pygame.Rect(VIEW_WIDTH // 2 - 110, VIEW_HEIGHT - 110, 220, 46)
    draw_button(surface, back, "Back", button_font, mouse_pos, (90, 140, 255))
    return {"back": back}


def draw_settings_screen(
    surface: pygame.Surface,
    fonts: dict,
    profile: dict,
    mouse_pos: Tuple[int, int],
) -> dict:
    surface.fill((8, 10, 16))
    title_font = fonts["title"]
    body_font = fonts["body"]
    button_font = fonts["button"]

    draw_centered(surface, title_font, "Settings", (245, 248, 252), (VIEW_WIDTH // 2, 58))
    draw_text(surface, body_font, "Left/Right: time limit", (210, 220, 235), (82, 120))
    draw_text(surface, body_font, "Up/Down: enemy density", (210, 220, 235), (82, 154))
    draw_text(surface, body_font, "Enter: save", (210, 220, 235), (82, 188))
    draw_text(surface, body_font, "Esc: back", (210, 220, 235), (82, 222))

    time_label = f"Time Limit: {int(profile['time_limit'])}s"
    enemy_label = f"Enemy Density: {profile['enemy_multiplier']:.1f}x"
    draw_centered(surface, body_font, time_label, (245, 248, 252), (VIEW_WIDTH // 2, 300))
    draw_centered(surface, body_font, enemy_label, (245, 248, 252), (VIEW_WIDTH // 2, 340))

    back = pygame.Rect(VIEW_WIDTH // 2 - 110, VIEW_HEIGHT - 110, 220, 46)
    draw_button(surface, back, "Back", button_font, mouse_pos, (90, 140, 255))
    return {"back": back}


def draw_end_screen(
    surface: pygame.Surface,
    fonts: dict,
    message: str,
    detail: str,
    mouse_pos: Tuple[int, int],
) -> dict:
    surface.fill((0, 0, 0))
    title_font = fonts["title"]
    body_font = fonts["body"]
    button_font = fonts["button"]
    draw_centered(surface, title_font, message, (245, 248, 252), (VIEW_WIDTH // 2, 240))
    draw_centered(surface, body_font, detail, (180, 190, 205), (VIEW_WIDTH // 2, 292))
    back = pygame.Rect(VIEW_WIDTH // 2 - 110, 372, 220, 46)
    draw_button(surface, back, "Main Menu", button_font, mouse_pos, (90, 140, 255))
    return {"back": back}


# ---------------------------
# Main loop
# ---------------------------
def main() -> None:
    random.seed(time.time_ns())
    load_env_file(".env")

    pygame.init()
    pygame.display.set_caption(SCREEN_TITLE)
    screen = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT))
    clock = pygame.time.Clock()

    tiles = load_tile_library()
    background = build_background_surface()
    title_font = pygame.font.Font(None, 42)
    body_font = pygame.font.Font(None, 24)
    credit_font = pygame.font.Font(None, 26)
    text_font = pygame.font.Font(None, 24)

    profile = load_profile()
    background_image = load_optional_surface(MENU_BG_PATH)
    title_image = load_optional_surface(MENU_TITLE_PATH)
    play_image = load_optional_surface(PLAY_BTN_PATH)
    customization_image = load_optional_surface(CUSTOMIZATION_BTN_PATH)
    exit_image = load_optional_surface(EXIT_BTN_PATH)
    player_image = load_optional_surface(PLAYER_IMAGE_PATH, (96, 96))
    vignette_overlay = build_vignette_overlay((VIEW_WIDTH, VIEW_HEIGHT))

    if not profile.get("name"):
        state = "name"
    else:
        state = "menu"

    fonts = {
        "title": title_font,
        "body": body_font,
        "button": pygame.font.Font(None, 30),
    }

    stream_config = load_world_stream_config()
    world: ChunkedWorld | None = None
    player: PlayerState | None = None
    camera: Camera | None = None
    name_buffer = profile.get("name", "")
    loading_needs_build = False
    loading_tip = LOADING_TIPS[0]
    loading_progress = 0.0
    end_message = ""
    end_detail = ""
    end_timer = 0.0

    running = True
    elapsed = 0.0

    while running:
        dt = min(clock.tick(60) / 1000.0, 1 / 30)
        elapsed += dt

        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = False
        start_game_requested = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            if state == "name":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        trimmed = name_buffer.strip()
                        if trimmed:
                            profile["name"] = trimmed[:24]
                            save_profile(profile)
                            state = "menu"
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_BACKSPACE:
                        name_buffer = name_buffer[:-1]
                    else:
                        if event.unicode and event.unicode.isprintable() and len(name_buffer) < 24:
                            name_buffer += event.unicode

            elif state == "menu":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        start_game_requested = True
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_clicked = True

            elif state == "customization":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        profile["color_index"] = (profile["color_index"] - 1) % len(PLAYER_COLORS)
                        save_profile(profile)
                    elif event.key == pygame.K_RIGHT:
                        profile["color_index"] = (profile["color_index"] + 1) % len(PLAYER_COLORS)
                        save_profile(profile)
                    elif event.key == pygame.K_i:
                        profile["use_player_image"] = not profile.get("use_player_image", False)
                        save_profile(profile)
                    elif event.key == pygame.K_RETURN:
                        save_profile(profile)
                        state = "menu"
                    elif event.key == pygame.K_ESCAPE:
                        state = "menu"
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_clicked = True

            elif state == "game":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                    start_game_requested = True
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    start_game_requested = True

            elif state in {"win", "lose"}:
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                    state = "menu"
                    end_timer = 0.0
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_clicked = True

        if state == "name":
            draw_name_entry_screen(screen, fonts, name_buffer, mouse_pos)
            if mouse_clicked:
                click_target = draw_name_entry_screen(screen, fonts, name_buffer, mouse_pos)
                if click_target["continue"].collidepoint(mouse_pos):
                    trimmed = name_buffer.strip()
                    if trimmed:
                        profile["name"] = trimmed[:24]
                        save_profile(profile)
                        state = "menu"
            pygame.display.flip()
            continue

        if state == "menu":
            buttons = draw_menu_screen(screen, fonts, profile, background_image, title_image, play_image, customization_image, exit_image, mouse_pos, elapsed)
            if mouse_clicked:
                if buttons["start"].collidepoint(mouse_pos):
                    start_game_requested = True
                elif buttons["customization"].collidepoint(mouse_pos):
                    state = "customization"
                elif buttons["exit"].collidepoint(mouse_pos):
                    running = False
            if start_game_requested:
                state = "loading"
                loading_needs_build = True
                loading_tip = LOADING_TIPS[int(elapsed) % len(LOADING_TIPS)]
                loading_progress = 0.0
            pygame.display.flip()
            continue

        if state == "customization":
            buttons = draw_customization_screen(screen, fonts, profile, player_image, mouse_pos, elapsed)
            if mouse_clicked and buttons["back"].collidepoint(mouse_pos):
                state = "menu"
            pygame.display.flip()
            continue

        if state == "loading":
            if loading_needs_build:
                draw_loading_screen(screen, title_font, body_font, loading_progress, "Building World", loading_tip)
                pygame.display.flip()
                pygame.event.pump()
                world = build_streaming_world(tiles, screen, title_font, body_font, stream_config)
                world.time_left = float(profile.get("time_limit", GAME_TIME_LIMIT))
                world.boxes_collected = 0
                world.lobby_returned = False
                world.game_over_reason = ""
                player = PlayerState(
                    x=world.spawn_world_x,
                    y=world.spawn_world_y,
                    color=PLAYER_COLORS[profile["color_index"]],
                )
                camera = Camera(x=world.spawn_world_x, y=world.spawn_world_y)
                elapsed = 0.0
                loading_needs_build = False
                state = "game"
            continue

        if state == "game":
            assert world is not None and player is not None and camera is not None

            shoot_pressed = start_game_requested
            keys = pygame.key.get_pressed()
            move_x = float((keys[pygame.K_d] or keys[pygame.K_RIGHT]) - (keys[pygame.K_a] or keys[pygame.K_LEFT]))
            move_y = float((keys[pygame.K_s] or keys[pygame.K_DOWN]) - (keys[pygame.K_w] or keys[pygame.K_UP]))
            submit_pressed = bool(keys[pygame.K_t])

            mouse_x, mouse_y = mouse_pos
            mouse_world_x = mouse_x + camera.x - VIEW_WIDTH / 2
            mouse_world_y = mouse_y + camera.y - VIEW_HEIGHT / 2
            player.facing = math.atan2(mouse_world_y - player.y, mouse_world_x - player.x)

            update_player(player, dt, move_x, move_y, world)
            update_streaming_world(world, player, dt, stream_config)
            world.time_left -= dt

            if shoot_pressed and len(world.bullets) < 64:
                world.bullets.append(bullet_spawn(player))

            update_enemies(world.enemies, player, dt, world)
            update_bullets(world.bullets, world.enemies, dt, world)
            collect_pickups(player, world.pickups)
            collect_boxes(player, world)
            collect_speed_items(player, world)
            update_lobby_objective(player, world, submit_pressed)

            camera_target_x = player.x + player.vx * CAMERA_LEAD
            camera_target_y = player.y + player.vy * CAMERA_LEAD
            smoothing = 1.0 - math.exp(-CAMERA_LAG * dt)
            camera.x += (camera_target_x - camera.x) * smoothing
            camera.y += (camera_target_y - camera.y) * smoothing
            world_w_px, world_h_px = world.world_bounds_px()
            camera.x = clamp(camera.x, VIEW_WIDTH / 2, world_w_px - VIEW_WIDTH / 2)
            camera.y = clamp(camera.y, VIEW_HEIGHT / 2, world_h_px - VIEW_HEIGHT / 2)

            if world.lobby_returned:
                end_message = "U Win!"
                end_detail = "Returned to the Lobby with all boxes."
                state = "win"
                end_timer = 0.0
            elif player.health <= 0:
                world.game_over_reason = "Health reached zero."
                end_message = "U Lose!"
                end_detail = world.game_over_reason
                state = "lose"
                end_timer = 0.0
            elif world.time_left <= 0:
                world.game_over_reason = "Time ran out."
                end_message = "U Lose!"
                end_detail = world.game_over_reason
                state = "lose"
                end_timer = 0.0

            draw_background(screen, background, camera)
            draw_world(screen, world, camera, 0, player, elapsed, player_image if profile.get("use_player_image", False) else None)
            screen.blit(vignette_overlay, (0, 0))
            draw_hud(screen, player, world, credit_font, text_font)
            pygame.display.flip()
            continue

        if state in {"win", "lose"}:
            end_timer += dt
            buttons = draw_end_screen(screen, fonts, end_message, end_detail, mouse_pos)
            if mouse_clicked and buttons["back"].collidepoint(mouse_pos):
                state = "menu"
                end_timer = 0.0
            if end_timer >= 1.5:
                state = "menu"
                end_timer = 0.0
            pygame.display.flip()
            continue

    pygame.quit()


if __name__ == "__main__":
    main()
