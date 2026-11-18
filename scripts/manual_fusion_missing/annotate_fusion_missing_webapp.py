#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import pandas as pd
from PIL import Image


WORD_TRANSLATION = {
    "red": "红色的", "blue": "蓝色的", "green": "绿色的", "yellow": "黄色的",
    "black": "黑色的", "white": "白色的", "purple": "紫色的", "orange": "橙色的",
    "pink": "粉色的", "brown": "棕色的", "gray": "灰色的", "grey": "灰色的",
    "striped": "条纹的", "spotted": "斑点的", "checkered": "格纹的",
    "wooden": "木制的", "metal": "金属的", "plastic": "塑料的", "glass": "玻璃的",
    "small": "小的", "large": "大的", "shiny": "闪亮的", "old": "旧的",
    "modern": "现代的", "furry": "毛茸茸的", "baby": "幼小的",
    "boy": "男孩", "girl": "女孩", "man": "男人", "woman": "女人", "person": "人",
    "dog": "狗", "cat": "猫", "bear": "熊", "rabbit": "兔子", "horse": "马",
    "elephant": "大象", "monkey": "猴子", "frog": "青蛙", "turtle": "乌龟",
    "bird": "鸟", "mouse": "老鼠", "lion": "狮子",
    "backpack": "背包", "car": "汽车", "chair": "椅子", "suitcase": "行李箱",
    "crown": "皇冠", "bench": "长椅", "clock": "时钟", "camera": "相机",
    "umbrella": "雨伞", "guitar": "吉他", "bicycle": "自行车", "bike": "自行车",
    "bowl": "碗", "bottle": "瓶子", "cup": "杯子", "table": "桌子",
    "lamp": "灯", "helmet": "头盔", "shoe": "鞋", "balloon": "气球",
    "kite": "风筝", "bus": "公交车", "train": "火车", "truck": "卡车",
    "boat": "船", "airplane": "飞机", "apple": "苹果", "banana": "香蕉",
    "strawberry": "草莓", "tomato": "番茄",
}

RELATIONS = [
    ("to the left of", "在{right}左边"),
    ("to the right of", "在{right}右边"),
    ("on the left of", "在{right}左边"),
    ("on the right of", "在{right}右边"),
    ("in front of", "在{right}前面"),
    ("next to", "在{right}旁边"),
    ("beside", "在{right}旁边"),
    ("near", "在{right}附近"),
    ("above", "在{right}上方"),
    ("below", "在{right}下方"),
    ("under", "在{right}下面"),
    ("over", "在{right}上方"),
    ("behind", "在{right}后面"),
    ("with", "和{right}"),
    ("and", "和{right}"),
]


def translate_np(x: str) -> str:
    x = str(x).strip(" .,:;").lower()
    x = re.sub(r"^(a|an|the)\s+", "", x)
    out = []
    for w in x.split():
        clean = w.strip(".,;:!?")
        if clean in {"a", "an", "the"}:
            continue
        out.append(WORD_TRANSLATION.get(clean, clean))
    return "".join(out)


def simple_translate_prompt(prompt: str) -> str:
    text = str(prompt).strip().lower()
    text = re.sub(r"\s+", " ", text)

    for rel_en, rel_zh in RELATIONS:
        if f" {rel_en} " in f" {text} ":
            left, right = text.split(rel_en, 1)
            return translate_np(left) + rel_zh.format(right=translate_np(right))

    words = re.sub(r"^(a|an|the)\s+", "", text).split()
    out = []
    for w in words:
        clean = w.strip(".,;:!?")
        if clean in {"a", "an", "the"}:
            continue
        if clean in {"and", "with"}:
            out.append("和")
        else:
            out.append(WORD_TRANSLATION.get(clean, clean))
    return "".join(out)


def safe_name(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(x))


def make_cache_image(src_path, dst_path, thumb_size=520, jpeg_quality=86):
    src_path = Path(str(src_path)).expanduser()
    if not src_path.is_absolute():
        src_path = (Path.cwd() / src_path).resolve()

    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists() and dst_path.stat().st_size > 0:
        return dst_path.resolve()

    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb_size, thumb_size))
            im.save(dst_path, format="JPEG", quality=jpeg_quality, optimize=True)
        return dst_path.resolve()
    except Exception as e:
        print(f"[WARN] image cache failed: {src_path}, error={e}", flush=True)
        placeholder = dst_path.parent / "placeholder_missing.jpg"
        if not placeholder.exists():
            img = Image.new("RGB", (520, 520), (80, 80, 80))
            img.save(placeholder, format="JPEG", quality=80)
        return placeholder.resolve()


def load_existing_annotations(out_csv: Path, annotator_id: str):
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return {}

    df = pd.read_csv(out_csv)
    if "annotator_id" not in df.columns or "pair_id" not in df.columns:
        return {}

    df = df[df["annotator_id"].astype(str) == str(annotator_id)]
    out = {}

    for _, r in df.iterrows():
        out[str(r["pair_id"])] = {
            "left_fusion": str(r.get("left_fusion", "0")),
            "left_missing": str(r.get("left_missing", "0")),
            "left_binding_error": str(r.get("left_binding_error", "0")),
            "left_object_correctness": str(r.get("left_object_correctness", "1")),
            "right_fusion": str(r.get("right_fusion", "0")),
            "right_missing": str(r.get("right_missing", "0")),
            "right_binding_error": str(r.get("right_binding_error", "0")),
            "right_object_correctness": str(r.get("right_object_correctness", "1")),
            "object_preference": str(r.get("object_preference", "Tie")),
            "notes": "" if pd.isna(r.get("notes", "")) else str(r.get("notes", "")),
        }
    return out


def save_or_update(out_csv: Path, row: dict):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and out_csv.stat().st_size > 0:
        df = pd.read_csv(out_csv)
        mask = ~(
            (df["annotator_id"].astype(str) == str(row["annotator_id"]))
            & (df["pair_id"].astype(str) == str(row["pair_id"]))
        )
        df = df[mask].copy()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df = df.sort_values(["pair_id"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")


HTML_TEMPLATE = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>SA-EBAMA Annotation</title>
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 16px 20px;
    background: #0b111c;
    color: #f8fafc;
    font-family: Arial, "Microsoft YaHei", sans-serif;
  }
  h1 {
    font-size: 26px;
    margin: 0 0 10px 0;
  }
  .top {
    font-size: 14px;
    line-height: 1.45;
    margin-bottom: 12px;
  }
  .top p { margin: 4px 0; }
  .toolbar {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 160px 1fr 1fr 1fr;
    gap: 12px;
    margin: 10px 0 14px 0;
    align-items: center;
  }
  button {
    height: 38px;
    border: 0;
    border-radius: 8px;
    background: #475569;
    color: white;
    font-weight: 700;
    font-size: 15px;
    cursor: pointer;
  }
  button:hover { background: #64748b; }
  .primary { background: #f97316; }
  .primary:hover { background: #fb923c; }
  input[type="number"] {
    height: 38px;
    border-radius: 8px;
    border: 1px solid #334155;
    background: #1e293b;
    color: white;
    padding: 0 10px;
    font-size: 15px;
    width: 100%;
  }
  .images {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 12px;
  }
  .img-card {
    background: #101827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    height: 345px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }
  .img-card img {
    max-height: 100%;
    max-width: 100%;
    object-fit: contain;
  }
  .score-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  .panel-title {
    font-weight: 700;
    margin: 4px 0 8px 0;
    font-size: 15px;
  }
  .score-panel {
    background: #111827;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px;
  }
  .score-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 8px;
  }
  .score-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px;
  }
  .score-label {
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 6px;
  }
  label {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-right: 10px;
    font-size: 14px;
    cursor: pointer;
  }
  input[type="radio"] { transform: scale(1.05); }
  .bottom {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 10px;
  }
  textarea {
    width: 100%;
    height: 42px;
    background: #1e293b;
    color: white;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px;
    resize: vertical;
  }
  .status {
    color: #93c5fd;
    font-weight: 700;
  }
  .small { color: #cbd5e1; }
</style>
</head>
<body>
  <h1>SA-EBAMA Fusion / Missing 盲标界面</h1>

  <div class="top">
    <p>
      <b>Annotator:</b> <span id="annotator"></span>
      &nbsp;&nbsp; <b>Status:</b> <span id="status" class="status"></span>
      &nbsp;&nbsp; <b>Progress:</b> <span id="progress"></span>
      &nbsp;&nbsp; <b>Done:</b> <span id="done"></span>
    </p>
    <p><b>Prompt:</b> <span id="prompt"></span></p>
    <p><b>中文提示词:</b> <span id="promptZh"></span></p>
    <p>
      <b>Dataset:</b> <span id="dataset"></span>
      &nbsp;&nbsp; <b>Pair ID:</b> <span id="pairId"></span>
      &nbsp;&nbsp; <b>Seed:</b> <span id="seed"></span>
    </p>
  </div>

  <div class="toolbar">
    <button onclick="prevItem()">上一张</button>
    <button onclick="nextItem()">下一张</button>
    <button onclick="firstUnfinished()">第一个未标注</button>
    <input id="jumpNo" type="number" min="1" value="1">
    <button onclick="jumpTo()">跳转</button>
    <button onclick="saveOnly()">保存</button>
    <button class="primary" onclick="saveAndNext()">保存并下一张</button>
  </div>

  <div class="images">
    <div class="img-card"><img id="leftImg"></div>
    <div class="img-card"><img id="rightImg"></div>
  </div>

  <div class="score-grid">
    <div>
      <div class="panel-title">Left Image 标注</div>
      <div class="score-panel">
        <div class="score-row">
          <div class="score-box"><div class="score-label">Fusion</div><div id="left_fusion"></div></div>
          <div class="score-box"><div class="score-label">Missing</div><div id="left_missing"></div></div>
        </div>
        <div class="score-row">
          <div class="score-box"><div class="score-label">Binding</div><div id="left_binding_error"></div></div>
          <div class="score-box"><div class="score-label">Correct</div><div id="left_object_correctness"></div></div>
        </div>
      </div>
    </div>

    <div>
      <div class="panel-title">Right Image 标注</div>
      <div class="score-panel">
        <div class="score-row">
          <div class="score-box"><div class="score-label">Fusion</div><div id="right_fusion"></div></div>
          <div class="score-box"><div class="score-label">Missing</div><div id="right_missing"></div></div>
        </div>
        <div class="score-row">
          <div class="score-box"><div class="score-label">Binding</div><div id="right_binding_error"></div></div>
          <div class="score-box"><div class="score-label">Correct</div><div id="right_object_correctness"></div></div>
        </div>
      </div>
    </div>
  </div>

  <div class="bottom">
    <div class="score-box">
      <div class="score-label">Object correctness preference，可选</div>
      <div id="object_preference"></div>
    </div>
    <div>
      <div class="score-label">Notes，可选</div>
      <textarea id="notes"></textarea>
    </div>
  </div>

<script>
const ANNOTATOR = __ANNOTATOR__;
const PAIRS = __PAIRS__;
let annotations = __ANNOTATIONS__;
let idx = 0;

const defaults = {
  left_fusion: "0",
  left_missing: "0",
  left_binding_error: "0",
  left_object_correctness: "1",
  right_fusion: "0",
  right_missing: "0",
  right_binding_error: "0",
  right_object_correctness: "1",
  object_preference: "Tie",
  notes: ""
};

function radioGroup(name, choices) {
  return choices.map(v => {
    return `<label><input type="radio" name="${name}" value="${v}"> ${v}</label>`;
  }).join("");
}

function buildRadios() {
  const score = ["0", "1", "2", "-1"];
  const correct = ["1", "0"];
  const pref = ["Left", "Right", "Tie", "Unclear"];

  for (const key of ["left_fusion", "left_missing", "left_binding_error", "right_fusion", "right_missing", "right_binding_error"]) {
    document.getElementById(key).innerHTML = radioGroup(key, score);
  }
  document.getElementById("left_object_correctness").innerHTML = radioGroup("left_object_correctness", correct);
  document.getElementById("right_object_correctness").innerHTML = radioGroup("right_object_correctness", correct);
  document.getElementById("object_preference").innerHTML = radioGroup("object_preference", pref);
}

function setRadio(name, value) {
  const el = document.querySelector(`input[name="${name}"][value="${value}"]`);
  if (el) el.checked = true;
}

function getRadio(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : defaults[name];
}

function doneCount() {
  return Object.keys(annotations).length;
}

function currentAnno() {
  const p = PAIRS[idx];
  return Object.assign({}, defaults, annotations[p.pair_id] || {});
}

function render() {
  const p = PAIRS[idx];
  const a = currentAnno();

  document.getElementById("annotator").textContent = ANNOTATOR;
  document.getElementById("status").textContent = annotations[p.pair_id] ? "已标注，可修改" : "未标注";
  document.getElementById("progress").textContent = `${idx + 1} / ${PAIRS.length}`;
  document.getElementById("done").textContent = `${doneCount()} / ${PAIRS.length}`;
  document.getElementById("prompt").textContent = p.prompt;
  document.getElementById("promptZh").textContent = p.prompt_zh;
  document.getElementById("dataset").textContent = p.dataset;
  document.getElementById("pairId").textContent = p.pair_id;
  document.getElementById("seed").textContent = p.seed;
  document.getElementById("jumpNo").value = idx + 1;

  document.getElementById("leftImg").src = p.left_img;
  document.getElementById("rightImg").src = p.right_img;

  for (const k of Object.keys(defaults)) {
    if (k === "notes") continue;
    setRadio(k, a[k]);
  }
  document.getElementById("notes").value = a.notes || "";
}

function collect() {
  return {
    pair_id: PAIRS[idx].pair_id,
    left_fusion: getRadio("left_fusion"),
    left_missing: getRadio("left_missing"),
    left_binding_error: getRadio("left_binding_error"),
    left_object_correctness: getRadio("left_object_correctness"),
    right_fusion: getRadio("right_fusion"),
    right_missing: getRadio("right_missing"),
    right_binding_error: getRadio("right_binding_error"),
    right_object_correctness: getRadio("right_object_correctness"),
    object_preference: getRadio("object_preference"),
    notes: document.getElementById("notes").value || ""
  };
}

async function saveOnly() {
  const data = collect();
  annotations[data.pair_id] = data;
  render();

  const resp = await fetch("/api/save", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data)
  });

  if (!resp.ok) {
    alert("保存失败，请查看服务器日志。");
  }
}

async function saveAndNext() {
  await saveOnly();
  if (idx < PAIRS.length - 1) {
    idx += 1;
    render();
  }
}

function prevItem() {
  if (idx > 0) {
    idx -= 1;
    render();
  }
}

function nextItem() {
  if (idx < PAIRS.length - 1) {
    idx += 1;
    render();
  }
}

function jumpTo() {
  const v = parseInt(document.getElementById("jumpNo").value, 10);
  if (!isNaN(v)) {
    idx = Math.max(0, Math.min(PAIRS.length - 1, v - 1));
    render();
  }
}

function firstUnfinished() {
  for (let i = 0; i < PAIRS.length; i++) {
    if (!annotations[PAIRS[i].pair_id]) {
      idx = i;
      render();
      return;
    }
  }
  idx = PAIRS.length - 1;
  render();
}

buildRadios();
render();
</script>
</body>
</html>
"""


class AppHandler(BaseHTTPRequestHandler):
    pairs_for_client = []
    pairs_internal = {}
    annotator_id = ""
    out_csv = None
    img_map = {}

    def log_message(self, fmt, *args):
        return

    def send_bytes(self, data: bytes, content_type: str, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_bytes(data, "application/json; charset=utf-8", status=status)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            annotations = load_existing_annotations(self.out_csv, self.annotator_id)

            html = HTML_TEMPLATE
            html = html.replace("__ANNOTATOR__", json.dumps(self.annotator_id, ensure_ascii=False))
            html = html.replace("__PAIRS__", json.dumps(self.pairs_for_client, ensure_ascii=False))
            html = html.replace("__ANNOTATIONS__", json.dumps(annotations, ensure_ascii=False))

            self.send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path.startswith("/img/"):
            name = urllib.parse.unquote(path.split("/img/", 1)[1])
            file_path = self.img_map.get(name)
            if not file_path or not Path(file_path).exists():
                self.send_bytes(b"not found", "text/plain; charset=utf-8", status=404)
                return

            data = Path(file_path).read_bytes()
            self.send_bytes(data, "image/jpeg")
            return

        self.send_bytes(b"not found", "text/plain; charset=utf-8", status=404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/save":
            self.send_json({"ok": False, "error": "not found"}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))

            pair_id = str(data["pair_id"])
            internal = self.pairs_internal[pair_id]

            row = {
                "annotator_id": self.annotator_id,
                "pair_id": pair_id,
                "dataset": internal["dataset"],
                "prompt_id": internal["prompt_id"],
                "prompt": internal["prompt"],
                "prompt_zh": internal["prompt_zh"],
                "seed": int(internal["seed"]),
                "left_method": internal["left_method"],
                "left_image_path": internal["left_image_path"],
                "right_method": internal["right_method"],
                "right_image_path": internal["right_image_path"],
                "left_fusion": int(data.get("left_fusion", 0)),
                "left_missing": int(data.get("left_missing", 0)),
                "left_binding_error": int(data.get("left_binding_error", 0)),
                "left_object_correctness": int(data.get("left_object_correctness", 1)),
                "right_fusion": int(data.get("right_fusion", 0)),
                "right_missing": int(data.get("right_missing", 0)),
                "right_binding_error": int(data.get("right_binding_error", 0)),
                "right_object_correctness": int(data.get("right_object_correctness", 1)),
                "object_preference": data.get("object_preference", "Tie"),
                "notes": data.get("notes", "") or "",
            }

            save_or_update(self.out_csv, row)
            self.send_json({"ok": True})
        except Exception as e:
            print("[ERROR] save failed:", e, flush=True)
            self.send_json({"ok": False, "error": str(e)}, status=500)


def prepare_data(args):
    pairs = pd.read_csv(args.pairs_csv).reset_index(drop=True)
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    pairs_for_client = []
    pairs_internal = {}
    img_map = {}

    print(f"[INFO] preparing {len(pairs)} pairs ...", flush=True)

    for i, r in pairs.iterrows():
        pair_id = str(r["pair_id"])
        pair_safe = safe_name(pair_id)

        left_name = f"{i+1:03d}_{pair_safe}_left.jpg"
        right_name = f"{i+1:03d}_{pair_safe}_right.jpg"

        left_file = make_cache_image(
            r["left_image_path"],
            cache_dir / left_name,
            thumb_size=args.thumb_size,
            jpeg_quality=args.jpeg_quality,
        )
        right_file = make_cache_image(
            r["right_image_path"],
            cache_dir / right_name,
            thumb_size=args.thumb_size,
            jpeg_quality=args.jpeg_quality,
        )

        prompt_zh = simple_translate_prompt(str(r["prompt"]))

        client_row = {
            "pair_id": pair_id,
            "dataset": str(r["dataset"]),
            "prompt_id": str(r["prompt_id"]),
            "prompt": str(r["prompt"]),
            "prompt_zh": prompt_zh,
            "seed": int(r["seed"]),
            "left_img": f"/img/{left_name}",
            "right_img": f"/img/{right_name}",
        }
        pairs_for_client.append(client_row)

        internal_row = {
            **r.to_dict(),
            "prompt_zh": prompt_zh,
        }
        pairs_internal[pair_id] = internal_row

        img_map[left_name] = str(left_file)
        img_map[right_name] = str(right_file)

        if (i + 1) % 10 == 0 or i + 1 == len(pairs):
            print(f"[INFO] cached {i + 1}/{len(pairs)}", flush=True)

    return pairs_for_client, pairs_internal, img_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_csv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--annotator_id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--cache_dir", default="results/manual_fusion_missing/webapp_cache")
    parser.add_argument("--thumb_size", type=int, default=520)
    parser.add_argument("--jpeg_quality", type=int, default=86)
    return parser.parse_args()


def main():
    args = parse_args()

    pairs_for_client, pairs_internal, img_map = prepare_data(args)

    AppHandler.pairs_for_client = pairs_for_client
    AppHandler.pairs_internal = pairs_internal
    AppHandler.img_map = img_map
    AppHandler.annotator_id = args.annotator_id
    AppHandler.out_csv = Path(args.out_csv)

    server = ThreadingHTTPServer((args.host, args.port), AppHandler)

    print(f"[INFO] annotator: {args.annotator_id}", flush=True)
    print(f"[INFO] local url: http://{args.host}:{args.port}", flush=True)
    print(f"[INFO] out_csv: {args.out_csv}", flush=True)

    server.serve_forever()


if __name__ == "__main__":
    main()
