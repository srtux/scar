# Poster Blueprint: Remember What Got Fixed

Event: IEEE CS NextGen: Juniors at Conference 2026 (AI & Robotics). Abstract due October 31, 2026. Poster judging on November 6, 2026 in Milpitas, CA, on a 48 x 36 inch tri-fold board (12 in, 24 in, 12 in panels).

## Abstract (171 words)

AI customer-service agents can keep a small notes file of lessons from past chats, but which chats deserve a spot? We propose Emotion-Shift Memory: save a lesson only from chats where a frustrated customer ended up satisfied, because only those chats contain both a mistake and the fix that worked. In 2,624 public agent runs (tau2-bench), 77% of failed runs produced no error message, so error logs cannot pick the lessons. On 600 human-labeled chats (WildFeedback), a free CPU model (Laya) found frustrated-to-satisfied chats with 59% F1, compared with 45% for a sentiment word list. In a controlled test where every condition used the same agent, rule writer, and 400-token memory, saving only fixed chats lowered repeated mistakes on unseen tasks from 44% with no memory to 32% (19 vs 26 of 59 moments; p = 0.065). It also had fewer repeats than saving every chat (41%) or only unresolved chats (37%), though a larger test is needed to confirm those gaps. The whole system is a plain text file, keyword matching, and 1,218 lines of Python.

## 55-second pitch

Picture an AI agent at an airline help desk. When it makes a mistake, there is usually no error message: in 2,624 runs we studied, 77% of the failed runs never logged an error. So how can it learn? We watch the customer's mood. If a customer got frustrated and then ended up satisfied, that chat holds both the mistake and the fix. We save one line from each of those chats into a plain text file called MEMORY.md: when this happens, avoid this, do that instead.

First we checked that a computer can spot that mood shift. On 600 real human chats, a free model that runs on a laptop found these chats best: 59% F1, versus 45% for a word list. Then we ran a fair test. Same AI, same rule writer, same 400-token memory; the only change was which chats we saved. Saving only the fixed chats cut repeated mistakes from 44% to 32%, fewer than saving every chat or only the unresolved ones. The whole thing is 1,218 lines of Python, and the next step is a bigger test.

## Judge questions

**Is the improvement real, or luck?** On the same 59 moments, our memory avoided 9 mistakes that the no-memory agent made and made 2 that it avoided. An exact McNemar test gives p = 0.065, just above the usual 0.05 bar. The ranking of memory types matches our idea, but those gaps are not statistically significant with 59 cases, which is why a bigger test is our next step.

**Why not just save every chat?** With the same rule writer and the same 400 tokens, saving every chat had 24 repeats versus 19 for fixed chats only. Lessons from unresolved chats often say "hand the customer to a human" (3 of the first 7 airline rules), which the store rules allow only when the agent truly cannot help. None of the first 6 fixed-chat rules say that.

**Why use mood instead of error logs?** 77% of failed runs (669 of 869) had no tool error at all, so an error log never shows those mistakes. The customer's reaction still does.

**Why did the agent test use Flash-Lite instead of Laya?** Agent chats include tool errors and a store policy, and Flash-Lite can read both. Laya only reads the customer's replies. Testing Laya inside the agent test is planned next.

**Could the agent have seen the answers?** No. Tasks are split into two folds by task ID, so memory is always built from different tasks than the test. Benchmark answer keys are removed from every past chat before any model reads it, and a unit test checks this. The rule writer sees only the chat transcript.

**What counts as a repeat?** Each test moment is the first state-changing step of a run where an earlier agent went wrong: either its tool call crashed, or the task allows no change at all. The new agent gets the same conversation up to that point. A repeat means its next move is also wrong: a forbidden change, the wrong change, or a claim that it already did something it did not do.

## Poster map

| Panel | Sections |
| :--- | :--- |
| Left (12 in) | 01 The problem (silent failures chart), 02 The idea (2x2 mood boxes), 03 How it works (5 steps), 04 A real human chat (VADER vs Laya scores) |
| Center (24 in) | Key numbers strip, 05 Main experiment (5-bar chart with p-values), 06 Detector comparison, 07 Learning curve, 08 Full results table |
| Right (12 in) | 09 Inside MEMORY.md (unresolved vs fixed rules), 10 One moment it now gets right, 11 Takeaways and limits |

## Rebuilding the poster

```bash
.venv/bin/python poster/build_poster.py
```

`build_poster.py` reads `results/esm_results.json` and counts the code lines itself, so the poster cannot drift from the results. Open `poster/poster.html` in Chrome and print to PDF; the page size is set to 48 x 36 inches. The example chat (section 04) and the "before and after" example (section 10) were picked for readability from the candidates that `run_experiment.py` saves in the results file (`wildfeedback_showcase.candidates` and `stage2_memory.rescue_examples`).

## Code size

| Part | Files | Lines |
| :--- | :--- | ---: |
| Experiment | `src/esm/*.py`, `run_experiment.py` | 1,218 |
| Tests | `tests/*.py` | 123 |
| Poster builder | `poster/build_poster.py` | 415 |
| Total | | 1,756 |
