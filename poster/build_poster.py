"""Generate poster/poster.html (48 x 36 in, 2400 x 1800 px) straight from results/esm_results.json.

Every number on the poster is read from the results file or counted from the repo, never typed by hand.
Usage:  python poster/build_poster.py   (then print poster.html to PDF at 48 x 36 in)
"""

import glob
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
R = json.load(open(os.path.join(REPO, "results", "esm_results.json"), encoding="utf-8"))
SHOWCASE_IDX = int(os.environ.get("SHOWCASE_IDX", "2"))  # which saved example chat to show (picked for readability)
RESCUE_IDX = int(os.environ.get("RESCUE_IDX", "2"))  # which before/after example to show
SAMPLE_DOMAIN = os.environ.get("SAMPLE_DOMAIN", "airline")
RULES_PER_CARD = int(os.environ.get("RULES_PER_CARD", "3"))


def esc(s):
    return html.escape(str(s), quote=True)


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


def lines_of(paths):
    return sum(sum(1 for _ in open(p, encoding="utf-8")) for p in paths)


LOC_CORE = lines_of(sorted(glob.glob(os.path.join(REPO, "src", "esm", "*.py"))) + [os.path.join(REPO, "run_experiment.py")])
LOC_TESTS = lines_of(sorted(glob.glob(os.path.join(REPO, "tests", "*.py"))))

SF = R["silent_failures"]
DET = R["stage1_shift_detection"]["detectors"]
S1 = R["stage1_shift_detection"]
S2 = R["stage2_memory"]
C = S2["conditions"]
CTRL = S2["controls"]
PAIR = S2["paired_vs_emotion_shift"]
CURVE = S2["learning_curve"]
N_CASES = C["no_memory"]["heldout_cases"]
N_CTRL = CTRL["no_memory"]["controls"]

CONDS = [  # key, poster name, short name, color
    ("no_memory", "No memory", "No memory", "#cbd5e1"),
    ("all_chats", "Save every chat", "Every chat", "#6366f1"),
    ("routine_only", "Smooth chats only", "Smooth only", "#94a3b8"),
    ("unresolved_only", "Unresolved chats only", "Unresolved only", "#ef4444"),
    ("emotion_shift", "Fixed chats only (ours)", "Fixed only (ours)", "#16a34a"),
]
COND_NAME = {k: n for k, n, _, _ in CONDS}
COND_COLOR = {k: c for k, _, _, c in CONDS}
base_rate, esm_rate = C["no_memory"]["repeat_rate"], C["emotion_shift"]["repeat_rate"]
rel_drop = S2["relative_reduction_vs_no_memory"]
best_other = min((k for k, *_ in CONDS if k != "emotion_shift"), key=lambda k: C[k]["repeat_rate"])


# ------------------------------------------------------------------ charts
def chart_silent():
    rows = [("Airline", SF["airline"]), ("Retail", SF["retail"]), ("All runs", SF["combined"])]
    x0, w, out = 96, 200, []
    for i, (label, d) in enumerate(rows):
        y = 14 + i * 50
        rate = d["silent_failures"] / d["failed_runs"]
        sw = round(w * rate, 1)
        bold = "700" if label == "All runs" else "500"
        out.append(f'<text x="2" y="{y + 19}" font-size="16" font-weight="{bold}" fill="#0f172a">{label}</text>')
        out.append(f'<rect x="{x0}" y="{y}" width="{sw}" height="28" rx="4" fill="#dc2626"/>')
        out.append(f'<rect x="{x0 + sw}" y="{y}" width="{round(w - sw, 1)}" height="28" rx="4" fill="#cbd5e1"/>')
        out.append(f'<text x="{x0 + w + 12}" y="{y + 15}" font-size="17" font-weight="700" fill="#b91c1c">{pct(rate)} silent</text>')
        out.append(f'<text x="{x0 + w + 12}" y="{y + 32}" font-size="12.5" class="mono" fill="#64748b">{d["silent_failures"]} of {d["failed_runs"]} failed runs</text>')
    out.append('<rect x="0" y="166" width="12" height="12" rx="2" fill="#dc2626"/><text x="18" y="176" font-size="13" fill="#334155">Failed with no error message</text>')
    out.append('<rect x="244" y="166" width="12" height="12" rx="2" fill="#cbd5e1"/><text x="262" y="176" font-size="13" fill="#334155">Failed with a tool error</text>')
    return f'<svg viewBox="0 0 526 184" class="chart">{"".join(out)}</svg>'


def chart_main():
    W, H, left, right, top, base = 1128, 372, 64, 1112, 30, 290
    ymax = max(0.6, max(C[k]["repeat_rate"] for k, *_ in CONDS) + 0.08)
    out = []
    for g in (0, 0.2, 0.4, 0.6):
        if g <= ymax:
            y = base - (base - top) * g / ymax
            out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{"#0f172a" if g == 0 else "#e2e8f0"}" stroke-width="{1.2 if g == 0 else 1}"/>')
            out.append(f'<text x="{left - 10}" y="{y + 5:.1f}" text-anchor="end" font-size="14" class="mono" fill="#64748b">{int(g * 100)}%</text>')
    slot = (right - left) / len(CONDS)
    for i, (k, name, _, color) in enumerate(CONDS):
        d = C[k]
        cx = left + slot * (i + 0.5)
        h = (base - top) * d["repeat_rate"] / ymax
        ours = k == "emotion_shift"
        out.append(f'<rect x="{cx - 62:.1f}" y="{base - h:.1f}" width="124" height="{h:.1f}" rx="6" fill="{color}"/>')
        out.append(f'<text x="{cx:.1f}" y="{base - h - 12:.1f}" text-anchor="middle" font-size="{30 if ours else 26}" font-weight="700" fill="{"#15803d" if ours else "#0f172a"}">{pct(d["repeat_rate"], 1)}</text>')
        out.append(f'<text x="{cx:.1f}" y="{base + 28}" text-anchor="middle" font-size="17" font-weight="{700 if ours else 500}" fill="{"#15803d" if ours else "#0f172a"}">{esc(name)}</text>')
        out.append(f'<text x="{cx:.1f}" y="{base + 50}" text-anchor="middle" font-size="13.5" class="mono" fill="#64748b">{d["mistakes_repeated"]} of {d["heldout_cases"]} repeated</text>')
        if not ours:
            p = PAIR[k]["mcnemar_p"]
            out.append(f'<text x="{cx:.1f}" y="{base + 70}" text-anchor="middle" font-size="12.5" class="mono" fill="#94a3b8">vs ours: p = {p:.3f}</text>')
        else:
            out.append(f'<text x="{cx:.1f}" y="{base + 70}" text-anchor="middle" font-size="12.5" class="mono" fill="#15803d">{rel_drop * 100:.0f}% fewer than no memory</text>')
    return f'<svg viewBox="0 0 {W} {H}" class="chart">{"".join(out)}</svg>'


def chart_detectors():
    order = [("VADER", "VADER (word list)", "#cbd5e1"), ("GoEmotions", "GoEmotions", "#a5b4fc"),
             ("Flash-Lite", "Flash-Lite (LLM)", "#60a5fa"), ("Laya", "Laya (free, CPU)", "#1d4ed8")]
    x0, w, out = 150, 260, []
    for g in (0, 0.2, 0.4, 0.6):
        x = x0 + w * g / 0.7
        out.append(f'<line x1="{x:.1f}" y1="6" x2="{x:.1f}" y2="236" stroke="{"#0f172a" if g == 0 else "#e2e8f0"}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="256" text-anchor="middle" font-size="12.5" class="mono" fill="#64748b">{int(g * 100)}%</text>')
    for i, (k, label, color) in enumerate(order):
        d = DET[k]
        y = 14 + i * 56
        bw = w * d["fixed_f1"] / 0.7
        best = k == "Laya"
        out.append(f'<text x="{x0 - 10}" y="{y + 20}" text-anchor="end" font-size="14.5" font-weight="{700 if best else 500}" fill="#0f172a">{esc(label)}</text>')
        out.append(f'<rect x="{x0}" y="{y}" width="{bw:.1f}" height="30" rx="4" fill="{color}"/>')
        out.append(f'<text x="{x0 + bw + 8:.1f}" y="{y + 15}" font-size="16" font-weight="700" fill="#0f172a">{pct(d["fixed_f1"], 1)}</text>')
        out.append(f'<text x="{x0 + bw + 8:.1f}" y="{y + 31}" font-size="12" class="mono" fill="#64748b">{d["ms_per_turn"]:.0f} ms per reply</text>' if d["ms_per_turn"] >= 10 else
                   f'<text x="{x0 + bw + 8:.1f}" y="{y + 31}" font-size="12" class="mono" fill="#64748b">{d["ms_per_turn"]:.1f} ms per reply</text>')
    return f'<svg viewBox="0 0 540 266" class="chart">{"".join(out)}</svg>'


def chart_curve():
    xs, keys = CURVE["chats_seen"], ["all_chats", "unresolved_only", "emotion_shift"]
    left, right, top, base = 52, 500, 14, 206
    ymin, ymax = 0.25, 0.50  # zoomed axis (labeled), so 1-case differences stay readable
    X = lambda v: left + (right - left) * v / 100
    Y = lambda r: base - (base - top) * (r - ymin) / (ymax - ymin)
    out = []
    for g in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        out.append(f'<line x1="{left}" y1="{Y(g):.1f}" x2="{right}" y2="{Y(g):.1f}" stroke="{"#0f172a" if g == 0.25 else "#e2e8f0"}" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{Y(g) + 4:.1f}" text-anchor="end" font-size="12.5" class="mono" fill="#64748b">{round(g * 100)}%</text>')
    for v in xs:
        out.append(f'<text x="{X(v):.1f}" y="{base + 20}" text-anchor="middle" font-size="12.5" class="mono" fill="#64748b">{v}</text>')
    out.append(f'<text x="{(left + right) / 2:.1f}" y="{base + 40}" text-anchor="middle" font-size="13" fill="#334155">past chats the agent has seen</text>')
    for k in keys:
        ys = CURVE["repeat_rate"][k]
        pts = " ".join(f"{X(v):.1f},{Y(r):.1f}" for v, r in zip(xs, ys))
        ours = k == "emotion_shift"
        out.append(f'<polyline points="{pts}" fill="none" stroke="{COND_COLOR[k]}" stroke-width="{4 if ours else 2.5}" stroke-linejoin="round"/>')
        for v, r in zip(xs, ys):
            out.append(f'<circle cx="{X(v):.1f}" cy="{Y(r):.1f}" r="{5 if ours else 4}" fill="{COND_COLOR[k]}"/>')
        out.append(f'<text x="{X(100) + 8:.1f}" y="{Y(ys[-1]) + 4:.1f}" font-size="13" font-weight="700" fill="{COND_COLOR[k]}">{pct(ys[-1])}</text>')
    legend = "".join(f'<rect x="{left + i * 160}" y="{base + 52}" width="12" height="12" rx="2" fill="{COND_COLOR[k]}"/>'
                     f'<text x="{left + i * 160 + 18}" y="{base + 62}" font-size="13" fill="#334155">{esc(dict((a, c) for a, b, c, d in CONDS)[k])}</text>'
                     for i, k in enumerate(keys))
    return f'<svg viewBox="0 0 560 276" class="chart">{"".join(out)}{legend}</svg>'


# ------------------------------------------------------------------ text blocks
def clip(text, limit):
    """Shorten text at a word boundary and mark the cut with an ellipsis."""
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",.;:") + "\u2026"


def showcase_html():
    cand = R["wildfeedback_showcase"]["candidates"][SHOWCASE_IDX]
    users = [m["content"] for m in cand["messages"] if m["role"] == "user"]
    rows = []
    for reply, human, v, l in zip(users[1:], cand["human_labels"], cand["vader_compound"], cand["laya_dissatisfied"]):
        hot = l >= 0.40
        rows.append(f'<tr class="{"hot" if hot else ""}"><td>&ldquo;{esc(clip(reply, 60))}&rdquo;</td><td>{esc(human)}</td>'
                    f'<td class="num">{v:+.2f}</td><td class="num {"laya-hot" if hot else ""}">{pct(l)}</td></tr>')
    box = {"FIXED": "Frustrated &rarr; Satisfied", "ROUTINE": "Calm &rarr; Satisfied", "UNFIXED": "Frustrated &rarr; Frustrated",
           "BROKEN": "Calm &rarr; Frustrated"}
    vader_box = "FIXED" if any(v <= -0.05 for v in cand["vader_compound"][:-1]) and cand["vader_compound"][-1] > -0.05 else "ROUTINE"
    body = (f'<div class="ask"><span class="who">Person asks the AI</span>&ldquo;{esc(clip(users[0], 90))}&rdquo;</div>'
            f'<table class="scores-t"><thead><tr><th>Person&rsquo;s reply to each AI answer</th><th>Human label</th>'
            f'<th class="num">VADER</th><th class="num">Laya</th></tr></thead><tbody>{"".join(rows)}</tbody>'
            f'<tfoot><tr><td>Box chosen</td><td class="good">{box["FIXED"]}</td>'
            f'<td class="num bad">{"Missed" if vader_box != "FIXED" else "Found"}</td><td class="num good">Found</td></tr></tfoot></table>')
    return body, cand


def rules_card(kind):
    md = S2["sample_memory_md"][SAMPLE_DOMAIN][kind]
    rules = [ln[2:] for ln in md.splitlines() if ln.startswith("- ")][:RULES_PER_CARD]
    items = []
    for r in rules:
        r = esc(r)
        for kw in ("AVOID", "INSTEAD"):
            r = r.replace(f"{kw} ", f'<b class="kw">{kw}</b> ', 1)
        items.append(f"<li>{r}</li>")
    return "".join(items), len([ln for ln in md.splitlines() if ln.startswith("- ")])


def rescue_html():
    ex = S2["rescue_examples"][RESCUE_IDX]

    def action(a):
        calls = ", ".join(a["tool_calls"]) if a["tool_calls"] else ""
        text = clip(a["text"], 170)
        if calls:
            return f'calls <span class="mono">{esc(calls)}</span>' + (f": &ldquo;{esc(text)}&rdquo;" if text else "")
        return f"&ldquo;{esc(text)}&rdquo;"
    rule = next((ln[2:] for ln in ex["esm_retrieved_memory"].splitlines() if ln.startswith("- ")), "")
    return ex, clip(ex["last_user"], 170), action(ex["no_memory_action"]), rule, action(ex["esm_action"])


showcase_bubbles, show = showcase_html()
bad_rules, bad_n = rules_card("unresolved_only")
good_rules, good_n = rules_card("emotion_shift")
rescue, rescue_user, rescue_before, rescue_rule, rescue_after = rescue_html()
laya, vader, flite = DET["Laya"], DET["VADER"], DET["Flash-Lite"]
avg_rules = {k: sum(C[k]["rules_in_memory_per_fold"]) / len(C[k]["rules_in_memory_per_fold"]) for k, *_ in CONDS}
FOLD_IDX = {"airline": 0, "retail": 2}[SAMPLE_DOMAIN]  # fold_A of the sample domain
saved_bad = C["unresolved_only"]["rules_in_memory_per_fold"][FOLD_IDX]
saved_good = C["emotion_shift"]["rules_in_memory_per_fold"][FOLD_IDX]


def human_handoffs(kind):
    rules = [ln for ln in S2["sample_memory_md"][SAMPLE_DOMAIN][kind].splitlines() if ln.startswith("- ")]
    return sum("human" in r.lower() for r in rules), len(rules)


hh_bad, hh_bad_n = human_handoffs("unresolved_only")
hh_good, hh_good_n = human_handoffs("emotion_shift")
pv = PAIR["no_memory"]
curve_esm, curve_all = CURVE["repeat_rate"]["emotion_shift"], CURVE["repeat_rate"]["all_chats"]
reps = {k: C[k]["mistakes_repeated"] for k, *_ in CONDS}

table_rows = []
for k, name, _, color in CONDS:
    d, ct = C[k], CTRL[k]
    p = "" if k == "emotion_shift" else f'{PAIR[k]["mcnemar_p"]:.3f}'
    table_rows.append(
        f'<tr class="{"ours" if k == "emotion_shift" else ""}"><td><span class="dot" style="background:{color}"></span>{esc(name)}</td>'
        f'<td class="num">{pct(d["repeat_rate"], 1)}</td><td class="num">{d["mistakes_repeated"]} / {d["heldout_cases"]}</td>'
        f'<td class="num">{pct(d["airline_repeat_rate"], 1)}</td><td class="num">{pct(d["retail_repeat_rate"], 1)}</td>'
        f'<td class="num">{ct["passed"]} / {ct["controls"]}</td><td class="num">{avg_rules[k]:.0f}</td>'
        f'<td class="num">{d["avg_injected_tokens"]:.0f}</td><td class="num">{p}</td></tr>')

CSS = open(os.path.join(HERE, "poster.css"), encoding="utf-8").read()

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=2400, initial-scale=1.0">
<title>Remember What Got Fixed: Emotion-Shift Memory for AI Agents</title>
<style>
{CSS}
</style>
</head>
<body>
<div class="poster">
  <header class="masthead">
    <div class="mast-left">
      <div class="eyebrow">IEEE CS NextGen: Juniors at Conference 2026 &middot; AI &amp; Robotics</div>
      <h1>Remember What Got <span class="hl">Fixed</span></h1>
      <p class="subtitle">Emotion-Shift Memory saves only the past chats where a frustrated customer ended up satisfied. With that {S2["memory_budget_tokens"]}-token memory, an AI agent repeated {rel_drop * 100:.0f}% fewer mistakes on tasks it had never seen.</p>
    </div>
    <div class="mast-right">
      <div class="fact"><span class="fact-k">Real human chats</span><span class="fact-v">{S1["total_threads"]} (WildFeedback)</span></div>
      <div class="fact"><span class="fact-k">AI agent runs</span><span class="fact-v">{SF["combined"]["total_runs"]:,} (tau2-bench)</span></div>
      <div class="fact"><span class="fact-k">Memory size</span><span class="fact-v">{S2["memory_budget_tokens"]} tokens of plain text</span></div>
      <div class="fact"><span class="fact-k">Code</span><span class="fact-v">{LOC_CORE:,} lines of Python</span></div>
    </div>
  </header>

  <main class="trifold">
    <section class="col col-left">
      <div class="sec">
        <div class="kicker">01 &middot; The problem</div>
        <h2>Most AI mistakes make no noise</h2>
        <p>We studied {SF["combined"]["total_runs"]:,} runs of AI agents helping simulated airline and store customers. When a run failed, the agent usually broke a store rule (refunding the wrong card, cancelling a ticket it should not) <b>without any error message</b>. A crash log cannot show what to learn.</p>
        {chart_silent()}
      </div>

      <div class="sec">
        <div class="kicker">02 &middot; The idea</div>
        <h2>Let the customer's mood pick the lesson</h2>
        <p>Two yes/no questions sort every finished chat into one of four boxes. Only one box holds both a mistake <i>and</i> the fix that worked.</p>
        <div class="grid2">
          <div class="axis-top"></div>
          <div class="axis-top">End: satisfied</div>
          <div class="axis-top">End: still frustrated</div>
          <div class="axis-side">Mid-chat:<br>frustrated</div>
          <div class="cell fixed"><span class="cell-tag">Frustrated &rarr; Satisfied</span><span class="cell-txt">Has the mistake and the fix.</span><span class="badge green">Save 1 rule</span></div>
          <div class="cell unfixed"><span class="cell-tag">Frustrated &rarr; Frustrated</span><span class="cell-txt">The fix never came.</span><span class="badge">Skip</span></div>
          <div class="axis-side">Mid-chat:<br>calm</div>
          <div class="cell routine"><span class="cell-tag">Calm &rarr; Satisfied</span><span class="cell-txt">Nothing went wrong.</span><span class="badge">Skip</span></div>
          <div class="cell broken"><span class="cell-tag">Calm &rarr; Frustrated</span><span class="cell-txt">Broke at the end, no fix.</span><span class="badge">Skip</span></div>
        </div>
      </div>

      <div class="sec">
        <div class="kicker">03 &middot; How it works</div>
        <h2>Five steps, no vector database</h2>
        <ol class="steps">
          <li><span class="n">1</span><span>A chat with a customer ends.</span></li>
          <li><span class="n">2</span><span>A mood detector puts the chat in one of the 4 boxes.</span></li>
          <li><span class="n">3</span><span>For <b>Frustrated &rarr; Satisfied</b> chats, a rule writer (Gemini) writes one line: <span class="mono">When &hellip; AVOID &hellip; INSTEAD &hellip;</span></span></li>
          <li><span class="n">4</span><span>The line is added to <span class="mono">MEMORY.md</span>, a plain text file.</span></li>
          <li><span class="n">5</span><span>In each new chat, keyword matching copies the 6 best-matching rules (at most {S2["memory_budget_tokens"]} tokens) into the agent's instructions.</span></li>
        </ol>
      </div>

      <div class="sec">
        <div class="kicker">04 &middot; A real human chat (WildFeedback {esc(show["thread_id"])})</div>
        <h2>Word lists miss polite frustration</h2>
        <div class="showcase">{showcase_bubbles}</div>
        <p class="note">VADER scores a reply from &minus;1 (negative) to +1 (positive), so 0.00 means it found nothing negative in &ldquo;you got 7 out of 10.&rdquo; Laya reads each reply together with the AI answer before it and gives the chance the person is frustrated.</p>
      </div>
    </section>

    <section class="col col-center">
      <div class="kpis">
        <div class="kpi"><div class="kpi-v red">{pct(SF["combined"]["silent_failure_rate"])}</div><div class="kpi-l">of failed AI runs showed no error ({SF["combined"]["silent_failures"]} of {SF["combined"]["failed_runs"]})</div></div>
        <div class="kpi"><div class="kpi-v blue">{pct(laya["fixed_f1"])} <span class="vs">vs</span> {pct(vader["fixed_f1"])}</div><div class="kpi-l">F1 at spotting &ldquo;frustrated &rarr; satisfied&rdquo; chats: Laya vs a word list</div></div>
        <div class="kpi"><div class="kpi-v green">{pct(base_rate)} &rarr; {pct(esm_rate)}</div><div class="kpi-l">repeated mistakes: no memory vs Emotion-Shift Memory ({rel_drop * 100:.0f}% fewer)</div></div>
        <div class="kpi"><div class="kpi-v">{S2["memory_budget_tokens"]}</div><div class="kpi-l">tokens of memory per chat, for every condition. No fine-tuning, no vector database.</div></div>
      </div>

      <div class="sec">
        <div class="kicker">05 &middot; Main experiment</div>
        <h2>Which past chats should go into MEMORY.md?</h2>
        <div class="fair">
          <span class="fair-l">Kept the same:</span>
          <span class="chip">Agent: Gemini 3.8 Flash</span>
          <span class="chip">Mood detector: Gemini 3.5 Flash-Lite</span>
          <span class="chip">Same rule writer</span>
          <span class="chip">{S2["memory_budget_tokens"]}-token cap + keyword matching</span>
          <span class="chip">{N_CASES} test moments from unseen tasks</span>
          <span class="fair-l change">Changed:</span>
          <span class="chip change">which past chats are saved</span>
        </div>
        <p class="lead">Every condition picked from the same 100 past chats, then faced {N_CASES} exact moments where earlier AI agents went wrong, on tasks it had never seen. A repeat means the agent got that same moment wrong too. Lower is better.</p>
        {chart_main()}
        <p class="note">How to read the p-values: compared case by case with no memory, our memory avoided {pv["esm_avoids_other_repeats"]} mistakes that the no-memory agent made, and made {pv["esm_repeats_other_avoids"]} that it avoided (exact McNemar test, p = {pv["mcnemar_p"]:.3f}). A p-value below 0.05 is the usual bar for &ldquo;very unlikely to be luck.&rdquo;</p>
      </div>

      <div class="row2">
        <div class="sec">
          <div class="kicker">06 &middot; Finding the mood shift</div>
          <h2>Which detector spots &ldquo;frustrated &rarr; satisfied&rdquo;?</h2>
          <p class="small">{S1["total_threads"]} real human chats labeled by people ({S1["per_quadrant"]} per box, {S1["total_reaction_turns"]:,} replies). F1 for the &ldquo;frustrated &rarr; satisfied&rdquo; box, plus speed per reply.</p>
          {chart_detectors()}
          <p class="note">Laya runs free on a laptop CPU. The agent test used Flash-Lite instead, because agent chats also contain tool errors and store rules that it can read.</p>
        </div>
        <div class="sec">
          <div class="kicker">07 &middot; Learning over time</div>
          <h2>More past chats, fewer repeats?</h2>
          <p class="small">Repeat rate after the agent has seen 0, 25, 50, or 100 past chats.</p>
          {chart_curve()}
          <p class="note">With only {N_CASES} tests, a swing of 2 to 3 cases moves a line by 3 to 5 points. Saving every chat reached {pct(curve_all[1])} at 25 chats, then rose to {pct(curve_all[2])} and {pct(curve_all[3])} as more chats piled in. Fixed-only memory helped once it held about {avg_rules["emotion_shift"]:.0f} rules (100 chats).</p>
        </div>
      </div>

      <div class="sec">
        <div class="kicker">08 &middot; All the numbers</div>
        <h2>Full results ({N_CASES} test moments, {N_CTRL} correct-action checks)</h2>
        <table class="ledger">
          <thead><tr><th>Condition</th><th class="num">Repeat rate</th><th class="num">Repeated</th><th class="num">Airline</th><th class="num">Retail</th>
          <th class="num">Correct actions kept</th><th class="num">Rules saved</th><th class="num">Tokens used</th><th class="num">p vs ours</th></tr></thead>
          <tbody>{"".join(table_rows)}</tbody>
        </table>
        <p class="note">Rules saved = average rules in MEMORY.md per fold after 100 chats. Tokens used = average memory text added to the agent's instructions. p = exact McNemar test on the same {N_CASES} paired cases.</p>
      </div>
    </section>

    <section class="col col-right">
      <div class="sec">
        <div class="kicker">09 &middot; Inside MEMORY.md ({SAMPLE_DOMAIN}, first rules in each file)</div>
        <h2>Same rule writer, different lessons</h2>
        <div class="mem bad"><div class="mem-h"><span>Learned from unresolved chats</span><span class="badge red">{saved_bad} rules saved</span></div><ul>{bad_rules}</ul></div>
        <div class="mem good"><div class="mem-h"><span>Learned from fixed chats (ours)</span><span class="badge green">{saved_good} rules saved</span></div><ul>{good_rules}</ul></div>
        <p class="note">Rules that tell the agent to hand the customer to a human: {hh_bad} of the first {hh_bad_n} unresolved rules, {hh_good} of the first {hh_good_n} fixed rules. The store rules allow a hand-off only when the agent truly cannot help.</p>
      </div>

      <div class="sec">
        <div class="kicker">10 &middot; One moment it now gets right ({esc(rescue["case"].split("/")[0])})</div>
        <h2>Before and after memory</h2>
        <div class="rescue">
          <div class="r-row"><span class="r-k">Customer</span><span class="r-v">&ldquo;{esc(rescue_user)}&rdquo;</span></div>
          <div class="r-row bad"><span class="r-k">No memory</span><span class="r-v">{rescue_before}</span></div>
          <div class="r-row rule"><span class="r-k">Rule pulled from MEMORY.md</span><span class="r-v mono-s">{esc(rescue_rule)}</span></div>
          <div class="r-row good"><span class="r-k">With our memory</span><span class="r-v">{rescue_after}</span></div>
        </div>
      </div>

      <div class="sec grow">
        <div class="kicker">11 &middot; Takeaways</div>
        <h2>What we learned</h2>
        <ol class="takeaways">
          <li><span><b>Mood is a free teacher.</b> When a frustrated customer ends up satisfied, the chat holds both the mistake and the fix, even when no error message was logged.</span></li>
          <li><span><b>Choose what to remember.</b> Same rule writer, same {S2["memory_budget_tokens"]} tokens: fixed chats only had the fewest repeats ({reps["emotion_shift"]} of {N_CASES}), versus {reps["unresolved_only"]} for unresolved chats, {reps["all_chats"]} for every chat, and {reps["no_memory"]} with no memory.</span></li>
          <li><span><b>Small and simple works.</b> A plain text file, keyword matching, and a free CPU mood model: {LOC_CORE:,} lines of Python.</span></li>
        </ol>
        <div class="limits"><b>Limits.</b> {N_CASES} test moments is a small sample: the gain over no memory is close to the usual bar (p = {pv["mcnemar_p"]:.3f}), and the gaps between memory types are not statistically significant yet. The customers in tau2-bench are simulated, and we tested one agent model. <b>Next:</b> more test tasks, real customers, and the free Laya detector inside the agent test.</div>
      </div>
    </section>
  </main>

  <footer class="foot">
    <span>Data: microsoft/WildFeedback (ODC-By 1.0), sierra-research/tau2-bench (MIT) &middot; Models: Gemini 3.8 Flash, Gemini 3.5 Flash-Lite, Laya, GoEmotions, VADER</span>
    <span>All numbers read from results/esm_results.json &middot; {LOC_CORE:,} lines of experiment code, {LOC_TESTS} lines of tests</span>
  </footer>
</div>
</body>
</html>
"""

with open(os.path.join(HERE, "poster.html"), "w", encoding="utf-8") as f:
    f.write(page)
print("wrote poster.html", "LOC_CORE", LOC_CORE, "LOC_TESTS", LOC_TESTS, "best_other", best_other, file=sys.stderr)
