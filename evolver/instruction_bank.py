# evolver/instruction_bank.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict

class InstructionBank:
    def __init__(self, base_instructions: List[str], save_path: Path):
        self.base = base_instructions
        self.save_path = save_path
        self.extra: List[str] = []
        self._load()

    def _load(self):
        if self.save_path.exists():
            try:
                data = json.loads(self.save_path.read_text(encoding="utf-8"))
                self.extra = data.get("extra_instructions", [])
            except Exception:
                self.extra = []

    def _save(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"extra_instructions": self.extra}
        self.save_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def size(self) -> int:
        return len(self.base) + len(self.extra)

    def get(self, idx: int) -> str:
        if idx < len(self.base):
            return self.base[idx]
        j = idx - len(self.base)
        return self.extra[j]

    def add_many(self, new_instrs: List[str]):
        # 去重+长度保护
        for s in new_instrs:
            s = (s or "").strip()
            if len(s) < 20:
                continue
            if s in self.base or s in self.extra:
                continue
            self.extra.append(s)
        self._save()
