# Per-tool freshness on TicToc

*Addendum to [NOTE.md](NOTE.md), draft of 8 October 2026. Same data and metric. These are exploratory analyses. As in
the note, Claude (Anthropic) implemented the analyses and drafted the text under my direction. Codex (OpenAI)
recomputed every number independently from the written premises, without reading the analysis implementations, and
reviewed this text and the script.*

The note argues that the right freshness interval depends on the tool. This addendum looks at how much that matters
on TicToc.

## 1. What a system knows beyond the clock

On TicToc the elapsed time alone matches most of the human labels. To see what a system adds beyond it, I score
balanced accuracy (NAR) inside narrow bins of elapsed time:
- the bins are half a decade of seconds wide;
- only bins that contain both labels count;
- each bin is weighted by its number of cases.

This is a diagnostic, not a new metric. The 30-minute rule scores 50% on these particular benchmark bins. Time can
still be used inside a bin, so this diagnostic does not remove every time-related difference between systems. On the
test split one bin, with 302 cases and only 4 "tool" labels, carries about 39% of the weight. With 26 test scenarios
the intervals are wide.

Test split, primary bins (1,069 cases; 1,067 for the two local models):

| System | NAR within mixed bins |
|---|---|
| 30-minute rule | 50.0% |
| Qwen3-8B in conversation, with timestamps | 55.9% |
| TTL class per scenario, inferred by gemma4 (frozen) | 56.9% |
| gemma4 e4b in conversation, with timestamps | 57.9% |
| TTL per scenario, fitted on the labels of the other conversations of the same test scenario | 77.7% |
| *In-sample references, using the test labels themselves:* | |
| *majority label of each scenario and time level* | *91.7%* |
| *best single threshold per scenario* | *96.6%* |

The fitted per-scenario TTL uses labels from the same test scenario. It is a reference for a well-calibrated per-tool
TTL, not a threshold learned on train and applied without test labels. Its target, set in advance, was 80%; it does
not reach it. Its advantage over the gemma4 classes has a 95% bootstrap interval over scenarios that includes zero
(−4.4 to +37.0 points).

## 2. The implied TTL depends on the tool

Where one threshold can separate a scenario's labels, every threshold between the oldest case labeled "direct" and the
youngest case labeled "tool" separates them equally well. The data constrain an interval, not a value. These
intervals describe the benchmark's preferences, not when the data actually change.

Across the 76 scenarios:
- 58 have such an interval;
- 3 have only "direct" labels, which give just a lower bound;
- in 15, no single threshold separates the labels.

The intervals span more than 4.85 orders of magnitude, or more than 5.15 when a scenario with no observed re-call is
included; that scenario only gives a lower bound.

Examples:
- live vehicle positions must switch before 93 seconds;
- the list of UN member states, after at least 76.8 days;
- programming-language syntax never switched within the tested ages, up to 154 days.

The benchmark has no case younger than 4 seconds, so it says nothing about sub-second freshness. One idea of mine,
not measured here: a host could ask for fresher data before an action with side effects, such as placing a stock
order, than before a plain answer.

## 3. Can a model declare the TTL?

I asked gemma4 e4b once per scenario, from the system prompt and the tool descriptions only: "after how many seconds
is that earlier result too old to reuse?". Host code then applied the number, with no tuning.

**Compared with the same model deciding inside the conversation.** On 1,067 common evaluable test cases:
- the declared TTL with the host's clock reaches NAR 84.87%;
- the model in conversation reaches 60.96%;
- the difference in global NAR is 23.91 points, with a 95% paired bootstrap interval over 26 scenarios of
  [12.0, 33.5].

This compares two procedures. They differ in prompt, number of calls, input, and in which component does the time
comparison, so it does not isolate any single cause. Inside the time bins of section 1, the advantage is uncertain
(−11.0 to +25.7 points).

**The declared numbers are poorly calibrated.** Of the 61 scenarios with an interval or a lower bound from section 2:
- 14 applied thresholds lie within the label-compatible intervals: 13 valid model declarations and one host fallback.
  Two of the 13 are "never" declarations, consistent only with lower bounds.
- 30 applied thresholds are too short and 17 too long. These include the other five fallbacks.

16 answers were "never", including ambulance dispatch. 6 answers were 0: the rule fixed in advance treated 0 as
invalid, and the host used 1,800 seconds instead.

## 4. Can a person declare it?

In an exploratory collection, one person gave thresholds for the 76 scenarios. Each scenario was presented in Italian
as a short plain-language description, with the question "after how long would you check this information again,
because it might have changed?". The answers and the texts shown are in `results/person_declared_ttl.json`, published
anonymously with the person's consent.

Applied by the host, these thresholds reach 91.27% within the time bins of section 1, against 71.10% for gemma4's
declarations: a difference of 20.17 points, with a 95% bootstrap interval over scenarios of [11.3, 34.6].
- **Against the intervals of section 2:** of the same 61 scenarios, 30 thresholds are inside, 9 too short and 22 too
  long.
- **Plain NAR:** 88.6%, below the 30-minute rule's 94.5%. The person's thresholds are often longer than the TicToc
  labels for data that change slowly. For company policies, for example, the person said a year, while the labels
  switch somewhere between 6 and 78 days.
- **"Never":** the person never answered it; gemma4 did 16 times.

Limits:
- **One person.** The comparison does not estimate how people in general would answer, and it does not show that the
  thresholds match when the data really change.
- **Blindness cannot be verified.** The person answered in the same chat where some results had been discussed, and I
  organized it knowing them. For example, the chat had said earlier that the ambulance switch happens at about 3
  minutes, and the person answered 3 minutes. The person also answered 3 minutes for emergency alerts, so this shows
  possible exposure, not that the message caused the answer.
- **Changes along the way.** After two answers showed the question was unclear, it was simplified and those two
  questions were asked again. Questions 12 to 76 were shown together in one list.
- **The texts are not equivalent to the originals.** The plain-language descriptions dropped some details, for
  example that a hotel booking fails if the same guest already booked the room. The wording, language and format also
  differ from gemma4's English prompt.

## What this suggests

- **A single default is a strong baseline on TicToc, not a freshness policy.**
- **Per-tool thresholds recover part of what the clock misses on this benchmark.** These diagnostics do not isolate
  why: the type of request, calibration choices and noisy preferences can all play a part.
- **Who declares the TTL matters.** A small model's declared numbers, applied by the host, beat the same model deciding
  in conversation in global NAR on the common evaluable cases, but they are poorly calibrated. One person's answers,
  with the limits above, come closer to the labels.

These results motivate testing server-provided freshness hints. Whether developers can calibrate those hints reliably
was not measured here.

## Reproduce

Run `python per_tool.py data/train.parquet data/test.parquet` from the root of this repository.
- The frozen gemma4 declarations are in `results/gemma4_declared_ttl.json`.
- `--declare` asks the model again and writes a new file, never the frozen one.
- `python -m unittest test_per_tool` runs the regression tests.
