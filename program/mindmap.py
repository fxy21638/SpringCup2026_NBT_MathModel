"""
用 Mermaid flowchart 语法 + kroki.io API 绘制问题分析图。

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


MERMAID_CODE = r"""flowchart LR
  subgraph left[" "]
    direction TB
    prep0["前期准备<br/>读取数据"]
    prep1["数据处理<br/>缺失异常处理"]
    prep2["模型求解<br/>分析问题"]
    prep0 --> prep1 --> prep2
  end

  prep2 --> hub(( ))

  hub --> q1["问题一"]
  hub --> q2["问题二"]
  hub --> q3["问题三"]

  subgraph row1[" "]
    direction LR
    q1 --> q1a["模糊综合评价"]
    q1a --> q1b["污染物指标"]
    q1a --> q1c["隶属度函数"]
    q1b --> q1d["熵权赋权"]
    q1c --> q1d
    q1d --> q1e["AQI 等级"]
  end

  subgraph row2[" "]
    direction LR
    q2 --> q2a["传感器预测<br/>使用问题一 AQI"]
    q2a --> q2b["传感器特征"]
    q2a --> q2c["XGBoost 模型"]
    q2b --> q2d["时序验证"]
    q2c --> q2d
    q2d --> q2e["预测结果"]
  end

  subgraph row3[" "]
    direction LR
    q3 --> q3a["改善建议<br/>结合评价与预测"]
    q3a --> q3b["特征重要性"]
    q3a --> q3c["污染源分析"]
    q3b --> q3d["治理建议"]
    q3c --> q3d
  end

  classDef prep fill:#ffffff,stroke:#263238,color:#263238,stroke-width:2px;
  classDef question fill:#ffffff,stroke:#263238,color:#263238,stroke-width:2px,font-weight:bold;
  classDef method fill:#ffffff,stroke:#374151,color:#263238,stroke-width:2px;
  classDef hidden fill:transparent,stroke:transparent,color:transparent;
  class prep0,prep1,prep2 prep;
  class q1,q2,q3 question;
  class q1a,q1b,q1c,q1d,q1e,q2a,q2b,q2c,q2d,q2e,q3a,q3b,q3c,q3d method;
  class hub hidden;
"""


THEME_CONFIG = {
    "theme": "base",
    "flowchart": {
        "htmlLabels": True,
        "curve": "linear",
        "nodeSpacing": 45,
        "rankSpacing": 65,
    },
    "themeVariables": {
        "fontFamily": "Microsoft YaHei, SimHei, Arial, sans-serif",
        "fontSize": "15px",
        "primaryColor": "#ffffff",
        "primaryTextColor": "#111827",
        "lineColor": "#374151",
        "clusterBkg": "#ffffff",
        "clusterBorder": "#ffffff",
    },
}


def render_via_kroki(code: str, output_path: Path, theme: dict | None = None) -> bool:
    """通过 kroki.io API 渲染 Mermaid 为 PNG/SVG。"""
    output_format = output_path.suffix.lstrip(".").lower()
    if output_format not in {"png", "svg"}:
        raise ValueError(f"不支持的输出格式: {output_path.suffix}")

    diagram_source = code
    if theme:
        diagram_source = f"%%{{init: {json.dumps(theme, ensure_ascii=False)}}}%%\n{code}"

    payload = json.dumps(
        {
            "diagram_source": diagram_source,
            "diagram_type": "mermaid",
            "output_format": output_format,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    req = urllib.request.Request(
        f"https://kroki.io/mermaid/{output_format}",
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
    output_mmd = OUTPUT_STEM.with_suffix(".mmd")
    output_png = OUTPUT_STEM.with_suffix(".png")
    output_svg = OUTPUT_STEM.with_suffix(".svg")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_mmd.write_text(MERMAID_CODE, encoding="utf-8")
    print(f"[OK] 源文件: {output_mmd}")

    ok_png = render_via_kroki(MERMAID_CODE, output_png, THEME_CONFIG)
    ok_svg = render_via_kroki(MERMAID_CODE, output_svg, THEME_CONFIG)

    return 0 if ok_png and ok_svg else 1


if __name__ == "__main__":
    raise SystemExit(main())
