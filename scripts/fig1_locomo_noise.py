"""Generate Figure 1: LoCoMo Noise Generation Pipeline diagram.

Produces a left-to-right 3-column layout showing:
  Column 1 — Original LoCoMo session with needle turns highlighted
  Column 2 — LLM noise generator with 3 noise types
  Column 3 — Resulting noisy session + QA evidence pointer

Output: locomo_noise_fig1.pdf in the repo root.
"""

import graphviz

dot = graphviz.Digraph("LoCoMo_Noise", format="pdf")
dot.attr(
    rankdir="LR",
    fontname="Helvetica",
    fontsize="11",
    bgcolor="white",
    pad="0.3",
    nodesep="0.4",
    ranksep="0.8",
)

# ── Column 1: Original LoCoMo Session ────────────────────────────────
with dot.subgraph(name="cluster_orig") as c:
    c.attr(
        label="Original LoCoMo Session",
        style="rounded",
        color="#333333",
        fontname="Helvetica-Bold",
        fontsize="12",
    )
    turns = [
        ("t1", 'A: "Hey Mel! How are you?"', False),
        ("t2", 'B: "Great! I just joined an\nLGBTQ+ support group"', True),
        ("t3", 'A: "That\'s wonderful!\nHow was it?"', False),
        ("t4", 'B: "Really welcoming.\nI felt accepted"', True),
        ("t5", 'A: "I\'m so happy for you!"', False),
    ]
    for tid, label, is_needle in turns:
        if is_needle:
            c.node(
                tid, label,
                shape="box", style="filled",
                fillcolor="#dae8fc", color="#6c8ebf", fontsize="9",
                fontname="Helvetica",
            )
        else:
            c.node(
                tid, label,
                shape="box", style="filled",
                fillcolor="#f5f5f5", color="#999999", fontsize="9",
                fontname="Helvetica",
            )
    for i in range(len(turns) - 1):
        c.edge(turns[i][0], turns[i + 1][0], style="invis")

# ── Column 2: Noise Generator ────────────────────────────────────────
with dot.subgraph(name="cluster_gen") as c:
    c.attr(
        label="LLM Noise Generator\n(GPT-4o-mini)",
        style="rounded,dashed",
        color="#666666",
        fontname="Helvetica-Bold",
        fontsize="12",
    )
    c.node(
        "nf", 'Filler (40%)\n"Yeah doing well!"',
        shape="note", style="filled",
        fillcolor="#fff2cc", color="#d6b656", fontsize="9",
        fontname="Helvetica",
    )
    c.node(
        "ns", 'Status (30%)\n"brb getting coffee"',
        shape="note", style="filled",
        fillcolor="#fff2cc", color="#d6b656", fontsize="9",
        fontname="Helvetica",
    )
    c.node(
        "nt", 'Tangent (30%)\n"Did you see the game?"',
        shape="note", style="filled",
        fillcolor="#fff2cc", color="#d6b656", fontsize="9",
        fontname="Helvetica",
    )
    c.edge("nf", "ns", style="invis")
    c.edge("ns", "nt", style="invis")

# ── Column 3: Noisy Result ───────────────────────────────────────────
with dot.subgraph(name="cluster_noisy") as c:
    c.attr(
        label="Noisy Session (75% noise ratio)",
        style="rounded",
        color="#333333",
        fontname="Helvetica-Bold",
        fontsize="12",
    )
    noisy = [
        ("r1",   'A: "Hey Mel! How are you?"',            "normal"),
        ("r_n1", 'B: "Yeah doing well, thanks!"',         "noise"),
        ("r2",   'B: "I just joined an\nLGBTQ+ support group"', "needle"),
        ("r_n2", 'A: "brb getting coffee"',               "noise"),
        ("r_n3", 'B: "Did you see the\ngame last night?"', "noise"),
        ("r3",   'A: "That\'s wonderful!"',               "normal"),
        ("r4",   'B: "Really welcoming.\nI felt accepted"', "needle"),
        ("r_n4", 'A: "Oh nice, same here"',               "noise"),
        ("r5",   'A: "I\'m so happy for you!"',           "normal"),
    ]
    for tid, label, typ in noisy:
        if typ == "needle":
            c.node(
                tid, label,
                shape="box", style="filled",
                fillcolor="#dae8fc", color="#6c8ebf", fontsize="9",
                fontname="Helvetica",
            )
        elif typ == "noise":
            c.node(
                tid, label,
                shape="box", style="filled,dashed",
                fillcolor="#fff2cc", color="#d6b656", fontsize="9",
                fontcolor="#888888", fontname="Helvetica",
            )
        else:
            c.node(
                tid, label,
                shape="box", style="filled",
                fillcolor="#f5f5f5", color="#999999", fontsize="9",
                fontname="Helvetica",
            )
    for i in range(len(noisy) - 1):
        c.edge(noisy[i][0], noisy[i + 1][0], style="invis")

# ── QA Evidence pointer ──────────────────────────────────────────────
dot.node(
    "qa",
    'QA: "When did B join\nthe support group?"\nEvidence → D1:2',
    shape="box", style="rounded,filled",
    fillcolor="#d5e8d4", color="#82b366", fontsize="9",
    fontname="Helvetica",
)
dot.edge(
    "r2", "qa",
    color="#82b366", style="bold",
    label="  evidence", fontsize="8", fontcolor="#82b366",
    fontname="Helvetica",
)

# ── Cross-cluster edges (pipeline flow) ──────────────────────────────
dot.edge(
    "t3", "nf",
    label="  inject\n  noise", style="dashed",
    color="#d6b656", fontsize="8", fontcolor="#666666",
    fontname="Helvetica",
)
dot.edge("nf", "r_n1", style="dashed", color="#d6b656")

# ── Legend ────────────────────────────────────────────────────────────
dot.node(
    "leg",
    '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
    '<TR>'
    '<TD BGCOLOR="#dae8fc">Needle (key fact)</TD>'
    '<TD BGCOLOR="#fff2cc"><FONT COLOR="#888888">Noise turn</FONT></TD>'
    '<TD BGCOLOR="#f5f5f5">Normal turn</TD>'
    '</TR></TABLE>>',
    shape="none", fontsize="8",
)

# ── Render ────────────────────────────────────────────────────────────
out = dot.render("locomo_noise_fig1", cleanup=True)
print(f"Figure saved to: {out}")
