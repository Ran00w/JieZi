#!/usr/bin/env python3
"""Replace image path prefix in JieZi-VQA.jsonl."""

import json

SRC = "/media/dpctc/4TB1/lr/JieZi-hf/JieZi-VQA.jsonl"
DST = "/media/dpctc/4TB1/lr/JieZi-hf/JieZi-VQA-ascend.jsonl"
OLD = "/media/dpctc/4TB1/lr"
NEW = "/data/lr/ms/jiezi/llm/dataset"

count = 0
with open(SRC, "r", encoding="utf-8") as fin, open(DST, "w", encoding="utf-8") as fout:
    for line in fin:
        obj = json.loads(line)
        obj["images"] = [p.replace(OLD, NEW, 1) for p in obj["images"]]
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        count += 1
        if count % 50000 == 0:
            print(f"processed {count} lines...")

print(f"done: {count} lines written to {DST}")
