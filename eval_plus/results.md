Qwen2.5-Coder-3B-Instruct
----
humaneval (base tests)
pass@1:	0.848
humaneval+ (base + extra tests)
pass@1:	0.799
mbpp (base tests)
pass@1:	0.746
mbpp+ (base + extra tests)
pass@1:	0.635


axolotl-ai-co/qwen2.5-3b-instruct -code-grpo
----
humaneval (base tests)
pass@1:	0.750
humaneval+ (base + extra tests)
pass@1:	0.701
mbpp (base tests)
pass@1:	0.754
mbpp+ (base + extra tests)
pass@1:	0.632

qwen2.5-3b-instruct
---
humaneval (base tests)
pass@1:	0.713
humaneval+ (base + extra tests)
pass@1:	0.671
mbpp (base tests)
pass@1:	0.733
mbpp+ (base + extra tests)
pass@1:	0.601

---
| Model | HumanEval (base tests) pass@1 | HumanEval+ (base + extra tests) pass@1 | MBPP (base tests) pass@1 | MBPP+ (base + extra tests) pass@1 |
|-------|-------------------------------|---------------------------------------|--------------------------|----------------------------------|
| Qwen2.5-Coder-3B-Instruct | 0.848 | 0.799 | 0.746 | 0.635 |
| axolotl-ai-co/qwen2.5-3b-instruct-code-grpo | 0.750 | 0.701 | 0.754 | 0.632 |
| qwen2.5-3b-instruct | 0.713 | 0.671 | 0.733 | 0.601 |