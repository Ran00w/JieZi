import json

SCRIPT_ORDER = ["甲骨文", "金文", "篆书", "隶书", "楷书", "草书"]

def build_stages(char, scripts_present):
    ext_map = {
        "㐭": {"甲骨文": "jpg", "金文": "jpg", "篆书": "jpg", "隶书": "jpg", "楷书": "png", "草书": "jpg"},
        "㱃": {"甲骨文": "jpg", "金文": "jpg", "篆书": "jpg", "隶书": "jpg", "楷书": "png", "草书": "jpg"},
        "冀": {"金文": "jpg", "篆书": "jpg", "隶书": "jpg", "楷书": "png", "草书": "jpg"},
        "䍩": {"甲骨文": "jpg", "金文": "jpg", "篆书": "jpg", "隶书": "jpg", "楷书": "png", "草书": "jpg"},
    }
    stages = []
    for s in SCRIPT_ORDER:
        if s in scripts_present:
            ext = ext_map.get(char, {}).get(s, "jpg")
            stages.append({"script": s, "image": f"images/evo/{char}_{s}.{ext}"})
    return stages

# Read source entries from JSONL
jsonl_path = "/media/dpctc/4TB1/lr/JieZi-hf/JieZi-VQA.jsonl"
entries = {}
with open(jsonl_path) as f:
    for line in f:
        d = json.loads(line)
        eid = d.get("id", "")
        if eid in ["㐭_隶书_3", "㱃_楷书_7",
                    "冀_篆书_0::vqa::evolution", "䍩_楷书_4::vqa::evolution"]:
            msgs = d.get("messages", [])
            entries[eid] = {
                "q": msgs[0]["content"] if len(msgs) > 0 else "",
                "a": msgs[1]["content"] if len(msgs) > 1 else "",
            }

examples = [
    {
        "id": "㐭_隶书_3",
        "image": "images/evo/㐭_main.jpg",
        "character": "㐭",
        "script": "隶书",
        "task": "evolution",
        "question": entries["㐭_隶书_3"]["q"],
        "answer": entries["㐭_隶书_3"]["a"],
        "stages": build_stages("㐭", ["甲骨文", "金文", "篆书", "隶书", "楷书", "草书"])
    },
    {
        "id": "㱃_楷书_7",
        "image": "images/evo/㱃_main.png",
        "character": "㱃",
        "script": "楷书",
        "task": "evolution",
        "question": entries["㱃_楷书_7"]["q"],
        "answer": entries["㱃_楷书_7"]["a"],
        "stages": build_stages("㱃", ["甲骨文", "金文", "篆书", "隶书", "楷书", "草书"])
    },
    {
        "id": "冀_篆书_0::vqa::evolution",
        "image": "images/evo/冀_main.jpg",
        "character": "冀",
        "script": "篆书",
        "task": "evolution",
        "question": entries["冀_篆书_0::vqa::evolution"]["q"],
        "answer": entries["冀_篆书_0::vqa::evolution"]["a"],
        "stages": build_stages("冀", ["金文", "篆书", "隶书", "楷书", "草书"])
    },
    {
        "id": "鬱_楷书_6::vqa::evolution",
        "image": "images/img_00072.png",
        "character": "鬱",
        "script": "楷书",
        "task": "evolution",
        "question": "请描述该字自初文以来各历史阶段字形的传承与变异。",
        "answer": "甲骨文从二人在林中采集，会林木茂盛之意；金文从林从手持矢，篆文从林从鬱省会意兼形声。隶变后楷书写作鬱，异体作鬰。现代简化字将下部复杂的“鬱”省改为“缶”，形成现行简体字形。",
        "stages": [
            {"script": "甲骨文", "image": "images/img_00013.jpg"},
            {"script": "金文", "image": "images/img_00073.jpg"},
            {"script": "篆书", "image": "images/img_00074.jpg"},
            {"script": "隶书", "image": "images/img_00075.jpg"},
            {"script": "楷书", "image": "images/img_00072.png"},
            {"script": "草书", "image": "images/img_00076.jpg"}
        ]
    },
    {
        "id": "䍩_楷书_4::vqa::evolution",
        "image": "images/evo/䍩_main.png",
        "character": "䍩",
        "script": "楷书",
        "task": "evolution",
        "question": entries["䍩_楷书_4::vqa::evolution"]["q"],
        "answer": entries["䍩_楷书_4::vqa::evolution"]["a"],
        "stages": build_stages("䍩", ["甲骨文", "金文", "篆书", "隶书", "楷书", "草书"])
    }
]

# Read existing vqa_data.json
with open("/media/dpctc/4TB1/lr/JieZi-test-tmp/data/vqa_data.json") as f:
    vqa_data = json.load(f)

# Replace L4
for i, level in enumerate(vqa_data["levels"]):
    if level["id"] == "l4":
        vqa_data["levels"][i] = {
            "id": "l4",
            "name": "Level 4: 历时演变分析",
            "short_name": "演变分析",
            "description": "梳理汉字从甲骨文到楷书的形体递变过程",
            "tasks": ["evolution", "analysis"],
            "examples": examples
        }
        break

with open("/media/dpctc/4TB1/lr/JieZi-test-tmp/data/vqa_data.json", "w") as f:
    json.dump(vqa_data, f, ensure_ascii=False, indent=2)

print(f"Updated L4: {len(examples)} examples")
for ex in examples:
    print(f"  {ex['character']} ({ex['script']}): {len(ex['stages'])} stages, q={len(ex['question'])}c a={len(ex['answer'])}c")
