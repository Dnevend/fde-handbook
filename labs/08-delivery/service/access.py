"""8.5 文档级权限：身份 → 角色 → 允许的 doc_id 集合。

这是演示用的简化 ACL，不是完整的权限系统：
- 身份来自 `X-User-ID`（生产应接 SSO/OIDC）；
- 角色在 `access.yaml` 里配置；
- 默认拒绝（anonymous 看不到任何文档）；
- `allow_all` 再 `deny_docs` 表达“内部可见，但事故报告/客户资料除外”。

真正的交付要把这份策略接到客户的组织架构/文档系统，并保留审计。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = Path(__file__).resolve().parent / "access.yaml"
AUDIT = ROOT / "benchmarks" / "delivery" / "audit.jsonl"
_LOCK = threading.Lock()


@dataclass(frozen=True)
class User:
    user_id: str
    roles: tuple[str, ...]
    authenticated: bool

    def as_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "roles": list(self.roles),
            "authenticated": self.authenticated,
        }


def question_fingerprint(question: str) -> dict:
    """审计只记长度和哈希，不记问题原文。"""
    return {
        "question_len": len(question),
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest()[:16],
    }


class AccessPolicy:
    def __init__(self, raw: dict, all_docs: set[str], path: Path | None = None) -> None:
        self.raw = raw or {}
        self.version = int(self.raw.get("version", 0))
        self.default_role = str(self.raw.get("default_role", "anonymous"))
        self.roles: dict = self.raw.get("roles") or {}
        self.users: dict = self.raw.get("users") or {}
        self.all_docs = set(all_docs)
        self.path = path
        self.warnings: list[str] = []
        self._validate()

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AccessPolicy":
        policy_path = Path(path or os.environ.get("ACCESS_POLICY", DEFAULT_POLICY))
        all_docs = cls._load_doc_ids()
        if not all_docs:
            raise RuntimeError(
                "找不到 corpus/sources.yaml 的 doc_id；8.5 的 ACL 需要语料清单，"
                "先按第五部分准备语料。"
            )
        if not policy_path.exists():
            return cls(
                {
                    "version": 0,
                    "default_role": "anonymous",
                    "roles": {"anonymous": {"allow_docs": []}},
                    "users": {},
                },
                all_docs,
                policy_path,
            )
        raw = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
        return cls(raw, all_docs, policy_path)

    def _validate(self) -> None:
        """deny 写错是 fail-open，必须直接报错；allow 写错是 fail-closed，只告警。"""
        unknown_allow: set[str] = set()
        unknown_deny: set[str] = set()
        for spec in self.roles.values():
            for doc in spec.get("allow_docs") or []:
                if str(doc) not in self.all_docs:
                    unknown_allow.add(str(doc))
            for doc in spec.get("deny_docs") or []:
                if str(doc) not in self.all_docs:
                    unknown_deny.add(str(doc))
        if unknown_deny:
            raise ValueError(f"access.yaml 的 deny_docs 含未知 doc_id：{sorted(unknown_deny)}")
        if unknown_allow:
            message = f"access.yaml 的 allow_docs 含未知 doc_id（已忽略）：{sorted(unknown_allow)}"
            self.warnings.append(message)
            warnings.warn(message, stacklevel=2)

    @staticmethod
    def _load_doc_ids() -> set[str]:
        sources = ROOT / "corpus" / "sources.yaml"
        if not sources.exists():
            return set()
        data = yaml.safe_load(sources.read_text(encoding="utf-8")) or {}
        return {str(item["id"]) for item in data.get("sources", []) if "id" in item}

    def authenticate(self, user_id: str | None) -> User:
        if not user_id or user_id not in self.users:
            return User("anonymous", (self.default_role,), False)
        entry = self.users.get(user_id) or {}
        roles = tuple(entry.get("roles") or [self.default_role])
        return User(user_id, roles, True)

    def allowed_docs(self, user: User) -> set[str] | None:
        """返回允许的 doc_id；None 表示不限制，空集表示什么都不能看。"""
        if not self.all_docs:
            return set()

        allow_all = False
        allowed: set[str] = set()
        denied: set[str] = set()
        for role in user.roles:
            spec = self.roles.get(role) or {}
            if spec.get("allow_all"):
                allow_all = True
            allowed.update(str(item) for item in (spec.get("allow_docs") or []))
            denied.update(str(item) for item in (spec.get("deny_docs") or []))

        if allow_all:
            allowed = set(self.all_docs)
        allowed -= denied
        # 只保留真实存在的 doc_id，避免策略里写了错字还以为放行了
        allowed &= self.all_docs
        return None if allowed == self.all_docs else allowed

    def summary(self) -> dict:
        return {
            "version": self.version,
            "path": self.path.name if self.path else None,
            "default_role": self.default_role,
            "users": sorted(self.users),
            "roles": sorted(self.roles),
            "docs": len(self.all_docs),
        }

    def audit(self, row: dict) -> None:
        AUDIT.parent.mkdir(parents=True, exist_ok=True)
        payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **row}
        with _LOCK:
            with AUDIT.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
