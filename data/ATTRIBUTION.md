# Dataset Provenance and Attribution

All benchmark datasets used in this repository come from publicly released open-data and open-source repositories.

## 1. tau2-bench (data/tau2/)
- Source: [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) (data/tau2/results/final/ and tools.py)
- License: MIT License
- Files (2,624 simulations total across 4 LLMs):
  - claude-3-7-sonnet-20250219_airline_default_gpt-4.1-2025-04-14_4trials.json (200 simulations across 50 tasks)
  - gpt-4.1-2025-04-14_airline_default_gpt-4.1-2025-04-14_4trials.json (200 simulations across 50 tasks)
  - gpt-4.1-mini-2025-04-14_airline_base_gpt-4.1-2025-04-14_4trials.json (200 simulations across 50 tasks)
  - o4-mini-2025-04-16_airline_default_gpt-4.1-2025-04-14_4trials.json (200 simulations across 50 tasks)
  - claude-3-7-sonnet-20250219_retail_default_gpt-4.1-2025-04-14_4trials.json (456 simulations across 114 tasks)
  - gpt-4.1-2025-04-14_retail_default_gpt-4.1-2025-04-14_4trials.json (456 simulations across 114 tasks)
  - gpt-4.1-mini-2025-04-14_retail_base_gpt-4.1-2025-04-14_4trials.json (456 simulations across 114 tasks)
  - o4-mini-2025-04-16_retail_default_gpt-4.1-2025-04-14_4trials.json (456 simulations across 114 tasks)
  - airline_tools.json and retail_tools.json (JSON tool declarations)

## 2. Microsoft WildFeedback (data/wildfeedback/)
- Source: [microsoft/WildFeedback](https://huggingface.co/datasets/microsoft/WildFeedback) (sat_dsat_annotation.json)
- License: Open Data Commons Attribution License (ODC-By v1.0)
- Usage: 600 balanced human-annotated multi-turn threads (150 per quadrant, 2,709 reaction turns).
