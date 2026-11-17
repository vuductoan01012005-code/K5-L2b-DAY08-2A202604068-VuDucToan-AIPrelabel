"""Sinh JSON nhãn để dán vào CVAT (Project -> Labels -> tab Raw) từ submission/taxonomy.json.

    python scripts/make_cvat_labels.py

Ghi cvat/labels_generated.json. Nhãn trong CVAT PHẢI trùng tên với taxonomy (script map theo tên lớp).
"""
from __future__ import annotations

import argparse
import json

from common import ROOT, die, load_taxonomy, rel

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46c8c8", "#f032e6", "#9a6324", "#808000"]


def _default_str(val) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    return "" if val is None else str(val)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    tax = load_taxonomy()
    labels = []
    for i, name in enumerate(tax.classes):
        attrs = []
        if name != tax.ignore:
            for a in tax.attrs.values():
                default_val = _default_str(a.get("default", ""))
                if a["type"] == "select":
                    vals = [str(v) for v in a.get("values", [])]
                    if not vals:
                        die(f"Thuộc tính select '{a['name']}' trong taxonomy.json cần values không rỗng.")
                else:
                    vals = [default_val]
                attrs.append({"name": a["name"], "mutable": False, "input_type": a["type"],
                              "default_value": default_val,
                              "values": vals})
        # type "any": cho phép cả rectangle (detection) và polygon (segmentation) trong cùng Project
        labels.append({"name": name, "color": "#808080" if name == tax.ignore else COLORS[i % len(COLORS)],
                       "type": "any", "attributes": attrs})
    text = json.dumps(labels, ensure_ascii=False, indent=2)
    out = ROOT / "cvat" / "labels_generated.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text + "\n", encoding="utf8")
    print(f"Đã ghi {rel(out)} (taxonomy {tax.version}, {len(labels)} nhãn). "
          "Dán nội dung vào CVAT: Project -> Labels -> Raw -> Save (hoặc Done).")


if __name__ == "__main__":
    main()
