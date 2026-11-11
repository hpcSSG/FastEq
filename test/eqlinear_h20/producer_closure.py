"""Freeze transitive project-local Python imports from the bytes actually copied."""
from __future__ import annotations

import ast
from pathlib import Path
import shutil

from common import Evidence


class ProducerClosure:
    def __init__(self, source_root, destination):
        self.source_root, self.destination = Path(source_root), Path(destination)

    def freeze(self, seeds):
        pending, copied = list(seeds), {}
        while pending:
            name = pending.pop()
            if name in copied:
                continue
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe producer source path")
            source, target = self.source_root / relative, self.destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied[name] = Evidence.sha256(target)
            if target.suffix != ".py":
                continue
            # Parse the frozen copy, so an owner's later edits cannot change the graph.
            tree = ast.parse(target.read_text(), filename=str(target))
            modules = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules.add(node.module)
                elif isinstance(node, ast.Import):
                    modules.update(alias.name for alias in node.names)
            for module in modules:
                candidate = module.replace(".", "/") + ".py"
                if (self.source_root / candidate).is_file() and candidate not in copied:
                    pending.append(candidate)
        # Every frozen project-local literal import must resolve inside the snapshot.
        return dict(sorted(copied.items()))
