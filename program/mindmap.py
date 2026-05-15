"""
用 Graphviz DOT + kroki.io API 绘制问题分析图。

布局参考“左侧准备流程 + 右侧问题树状展开”的论文技术路线图。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.request


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output" / "picture"
OUTPUT_STEM = OUTPUT_DIR / "mindmap_problem_analysis"


DOT_CODE = r"""digraph G {
  graph [
    rankdir=LR,
    splines=ortho,
    bgcolor="white",
    pad="0.25",
    nodesep="0.55",
    ranksep="0.75"
  ];

  node [
    shape=box,
    style="rounded",
    color="#263238",
    penwidth=2,
    fontname="Microsoft YaHei",
    fontsize=14,
    margin="0.18,0.12",
    width=1.35,
    height=0.62
  ];

  edge [
    color="#374151",
    penwidth=1.5,
    arrowsize=0.7,
    fontname="Microsoft YaHei"
  ];

  prep0 [label="前期准备\n读取数据"];
  prep1 [label="数据处理\n缺失异常处理"];
  prep2 [label="模型求解\n分析问题"];

  prep0 -> prep1 [constraint=false];
  prep1 -> prep2 [constraint=false];

  hub [label="", shape=point, width=0.05, height=0.05, color="#374151"];
  prep2 -> hub;

  q1 [label="问题一", penwidth=2.3];
  q2 [label="问题二", penwidth=2.3];
  q3 [label="问题三", penwidth=2.3];
  hub -> q1;
  hub -> q2;
  hub -> q3;

  q1a [label="模糊综合评价"];
  q1b [label="污染物指标"];
  q1c [label="隶属度函数"];
  q1d [label="熵权赋权"];
  q1e [label="AQI 等级"];
  q1 -> q1a -> q1b -> q1d -> q1e;
  q1a -> q1c -> q1d;

  q2a [label="传感器预测\n使用问题一 AQI"];
  q2b [label="传感器特征"];
  q2c [label="XGBoost 模型"];
  q2d [label="时序验证"];
  q2e [label="预测结果"];
  q2 -> q2a -> q2b -> q2d -> q2e;
  q2a -> q2c -> q2d;

  q3a [label="改善建议\n结合评价与预测"];
  q3b [label="特征重要性"];
  q3c [label="污染源分析"];
  q3d [label="治理建议"];
  q3 -> q3a -> q3b -> q3d;
  q3a -> q3c -> q3d;

  { rank=same; prep0; prep1; prep2; }
}
"""


def render_via_kroki(code: str, output_path: Path) -> bool:
    """通过 kroki.io API 渲染 Graphviz DOT 为 PNG/SVG。"""
    output_format = output_path.suffix.lstrip(".").lower()
    if output_format not in {"png", "svg"}:
        raise ValueError(f"不支持的输出格式: {output_path.suffix}")

    payload = json.dumps(
        {
            "diagram_source": code,
            "diagram_type": "graphviz",
            "output_format": output_format,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    req = urllib.request.Request(
        f"https://kroki.io/graphviz/{output_format}",
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if resp.status == 200:
                img_data = resp.read()
                min_size = 200 if output_format == "svg" else 2000
                if len(img_data) > min_size:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(img_data)
                    print(f"[OK] 已渲染: {output_path} ({len(img_data)} bytes)")
                    return True
                print(f"[ERR] 输出文件太小: {len(img_data)} bytes")
    except Exception as e:
        print(f"[ERR] kroki.io 请求失败: {e}")
    return False


def main() -> int:
    output_dot = OUTPUT_STEM.with_suffix(".dot")
    output_png = OUTPUT_STEM.with_suffix(".png")
    output_svg = OUTPUT_STEM.with_suffix(".svg")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_dot.write_text(DOT_CODE, encoding="utf-8")
    print(f"[OK] 源文件: {output_dot}")

    ok_png = render_via_kroki(DOT_CODE, output_png)
    ok_svg = render_via_kroki(DOT_CODE, output_svg)

    return 0 if ok_png and ok_svg else 1


if __name__ == "__main__":
    raise SystemExit(main())
