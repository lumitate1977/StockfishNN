from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
INPUT_CHANNELS = 18
ACTION_SIZE = 5 * 64 * 64
PROMOTION_TO_SLOT = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}


def load_config() -> dict[str, Any]:
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            raise ValueError("config.json must contain a JSON object")
        return config
    except FileNotFoundError:
        raise RuntimeError(f"Missing config file: {CONFIG_PATH}") from None
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse config.json: {exc}") from exc


def resolve_stockfish_path(config: dict[str, Any]) -> Path:
    configured = str(config.get("stockfish_path", "")).strip().strip('"')
    candidates: list[str] = []
    if configured:
        candidates.append(configured)
    env_path = os.environ.get("STOCKFISH_PATH", "").strip().strip('"')
    if env_path:
        candidates.append(env_path)

    for name in ("stockfish.exe", "stockfish"):
        local = ROOT / name
        if local.is_file():
            candidates.append(str(local))
    in_path = shutil.which("stockfish") or shutil.which("stockfish.exe")
    if in_path:
        candidates.append(in_path)

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file():
            return path.resolve()
    searched = "\n".join(f"  - {item}" for item in candidates) or "  - no path configured"
    raise FileNotFoundError(
        "Stockfish executable not found. Download it from https://stockfishchess.org/download/ "
        "and set its full path in config.json (stockfish_path).\nChecked:\n" + searched
    )


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a position as 18 binary 8x8 planes."""
    planes = np.zeros((INPUT_CHANNELS, 8, 8), dtype=np.uint8)
    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        color_offset = 0 if piece.color == chess.WHITE else 6
        planes[color_offset + piece.piece_type - 1, row, col] = 1

    rights = [
        (chess.WHITE, True),
        (chess.WHITE, False),
        (chess.BLACK, True),
        (chess.BLACK, False),
    ]
    for index, (color, kingside) in enumerate(rights):
        has_right = (
            board.has_kingside_castling_rights(color)
            if kingside
            else board.has_queenside_castling_rights(color)
        )
        if has_right:
            planes[12 + index, :, :] = 1

    if board.turn == chess.WHITE:
        planes[16, :, :] = 1
    if board.ep_square is not None:
        row = 7 - chess.square_rank(board.ep_square)
        col = chess.square_file(board.ep_square)
        planes[17, row, col] = 1
    return planes


def move_to_index(move: chess.Move) -> int:
    slot = PROMOTION_TO_SLOT.get(move.promotion)
    if slot is None:
        raise ValueError(f"Unsupported promotion piece in move: {move}")
    return slot * 4096 + move.from_square * 64 + move.to_square


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class ChessNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = 48
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(ResidualBlock(channels), ResidualBlock(channels))
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 8 * 8, 256),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Linear(256, ACTION_SIZE)
        self.value_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(self.residual(self.stem(x)))
        return self.policy_head(features), self.value_head(features).squeeze(-1)


def choose_move(
    model: ChessNet,
    board: chess.Board,
    device: torch.device,
    temperature: float = 0.0,
) -> chess.Move:
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("No legal moves in this position")

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(encode_board(board).astype(np.float32)).unsqueeze(0).to(device)
        logits, _ = model(x)
        indices = torch.tensor([move_to_index(move) for move in legal_moves], device=device)
        legal_logits = logits[0, indices]
        if temperature <= 0:
            chosen = int(torch.argmax(legal_logits).item())
        else:
            scaled = legal_logits / max(temperature, 1e-3)
            probabilities = torch.softmax(scaled, dim=0)
            chosen = int(torch.multinomial(probabilities, 1).item())
    return legal_moves[chosen]


def load_model(
    checkpoint_path: str | Path | None = None,
    device: torch.device | None = None,
) -> tuple[ChessNet, torch.device, dict[str, Any]]:
    config = load_config()
    path = Path(checkpoint_path or config.get("model_path", "models/latest.pt"))
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f"No trained model found at {path}. Run train.bat (or python train.py) first."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ChessNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    metadata = checkpoint.get("metadata", {})
    return model, device, metadata
