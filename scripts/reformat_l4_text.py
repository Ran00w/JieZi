import json

with open("/media/dpctc/4TB1/lr/JieZi-test-tmp/data/vqa_data.json") as f:
    data = json.load(f)

l4 = [l for l in data["levels"] if l["id"] == "l4"][0]

updates = {
    "㐭_隶书_3": {
        "question": "请以专业文字学视角，分析该字的形体构造，并梳理其从造字之初到今日的字形演变历程。",
        "answer": (
            "【字头与结构】该隶书字形对应之现代字典字头为「㐭」，"
            "结构类型为⿱（上下结构），造字法归属象形字。\n\n"
            "【构件分解】该字由两个构件组成：\n"
            "「亠」承担表意功能，象粮仓的顶部；\n"
            "「回」承担表意功能，象粮仓的主体。\n\n"
            "【本义】据构形分析，该字初文本义为“容纳谷物的粮仓”。\n\n"
            "【字形演变】甲骨文象简易的粮仓之形；金文大同；篆文稍加简省并整齐化；"
            "隶变后笔画趋于平直；楷书写作“㐭”。"
            "该字是“稟”（禀）、“廩”（廪）的本字。"
        )
    },
    "㱃_楷书_7": {
        "question": "请从形体构造与历史演变两个角度，对该字进行专业的文字学分析。",
        "answer": (
            "【字头与结构】该楷书字形对应之现代字典字头为「㱃」，"
            "结构类型为⿰（左右结构），造字法归属会意字。\n\n"
            "【构件分解】该字由两个构件组成：\n"
            "「酓」承担表意功能，由甲骨文中酒坛及口舌之形演变而来，金文中讹变为从今从酉；\n"
            "「欠」承担表意功能，由甲骨文中张口伸舌的人形演变而来。\n\n"
            "【本义】据构形分析，该字初文本义为“喝”。\n\n"
            "【字形演变】甲骨文象一人张口伸舌就坛子饮酒之状；金文中口与舌脱离人形讹为“今”，"
            "人形讹变为“欠”；隶变后楷书写作“㱃”。该字是“饮”的初文。"
        )
    },
    # Keep 冀 original - it's concise and fine
    "䍩_楷书_4::vqa::evolution": {
        "question": "该字在汉字发展史上经历了怎样的形体演变？",
        "answer": (
            "【初文构造】甲骨文由“羊”与“支”构成，会意兼形声。\n\n"
            "【字形演变】经过隶变与楷化，右侧的“支”讹变为“攵”，最终形成“䍩”。"
            "该字被视为“養”（养）的古字异体。"
        )
    },
}

for ex in l4["examples"]:
    eid = ex["id"]
    if eid in updates:
        ex["question"] = updates[eid]["question"]
        ex["answer"] = updates[eid]["answer"]
        print(f"Updated: {ex['character']}")

with open("/media/dpctc/4TB1/lr/JieZi-test-tmp/data/vqa_data.json", "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Done")
