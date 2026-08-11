# neuro-code-math-dataset

A massive JSONL training dataset for LLM fine-tuning spanning **four domains**:

1. **Applied Neuroscience** — 40 subtopics (real research areas)
2. **Rust** — 35 subtopics (ownership, traits, async, etc.)
3. **Python** — 35 subtopics (generators, decorators, asyncio, etc.)
4. **Mathematics** — 40 subtopics (algebra, calculus, linear algebra, probability, statistics, discrete, etc.)

## Scale

| Metric | Value |
|---|---|
| Files | **150** JSONL files (one per subtopic) + `index.jsonl` |
| Entries | **1,125,000** chat-formatted examples |
| Approx. tokens | **~869 million** |
| Size on disk | **~3.4 GB** uncompressed |
| Format | `{"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}` |

## Domains & subtopic counts

| Domain | Files | Entries | Approx. tokens | Size |
|---|---|---|---|---|
| applied_neuroscience | 40 | 300,000 | ~248M | 983 MB |
| rust | 35 | 262,500 | ~199M | 794 MB |
| python | 35 | 262,500 | ~195M | 778 MB |
| math | 40 | 300,000 | ~227M | 901 MB |

## File naming

```
data/{domain}__{subtopic}.jsonl
```

Examples:
- `data/applied_neuroscience__neural_coding.jsonl`
- `data/rust__ownership.jsonl`
- `data/python__asyncio.jsonl`
- `data/math__eigenvalues_eigenvectors.jsonl`

## Entry schema

Each line is a JSON object:

```json
{
  "messages": [
    {"role": "user", "content": "Explain ownership in Rust for a beginner..."},
    {"role": "assistant", "content": "In Rust, **ownership** is the rule that..."}
  ]
}
```

This is the standard chat format used by OpenAI, Mistral, Llama 3, and many other fine-tuning frameworks.

## Variation strategy

Each subtopic produces 7,500 unique entries via a combinatorial variation matrix:

- **15 prompt templates** per domain (60 total)
- **10 audiences** (beginner, intermediate, expert, student, professional, teacher, self-learner, manager, researcher, reviewer)
- **10 perspectives** (theoretical, practical, historical, comparative, critical, teach, everyday, professional, debugging, optimization)
- **10 formats** (step-by-step, checklist, worked example, scenario, Q&A, compare-contrast, decision tree, common mistakes, before/after, principles-then-application)
- **5 opener variants** per domain
- **Permuted ordering** of the 5 principles, 3 examples, and 5 mistakes per subtopic

Theoretical maximum per subtopic: 15 × 10 × 10 × 10 × 5 × 5! × 3! × 5! = ~1.3 billion unique entries — far above the 7,500 target. Dedup is on the full entry (user + assistant) hash.

## Subtopic lists

### Applied Neuroscience (40)
neural_coding, neural_decoding, population_coding, temporal_coding, rate_coding, sparse_coding, predictive_coding, bayesian_brain, free_energy_principle, active_inference, global_workspace_theory, integrated_information_theory, higher_order_theories, predictive_processing, hierarchical_processing, cortical_minicolumns, thalamic_reticular_nucleus, basal_ganglia_circuits, cerebellar_computation, hippocampal_formation, entorhinal_grid_cells, place_cells, head_direction_cells, theta_oscillations, gamma_oscillations, cross_frequency_coupling, sharp_wave_ripples, sleep_replay, consolidation, long_term_potentiation, long_term_depression, spike_timing_dependent_plasticity, homeostatic_plasticity, metaplasticity, dendritic_computation, axonal_computation, neuromodulation, dopamine_reward, serotonin_modulation, neural_manifolds

### Rust (35)
ownership, borrowing, lifetimes, mutable_references, slice_types, strings_str_vs_string, traits, trait_objects, generic_types, where_clauses, associated_types, default_impls, trait_bounds, newtype_pattern, phantom_types, drop_trait, deref_trait, from_into_traits, iterators, iterator_adapters, collect_into, async_await, futures, pinning, tokio_runtime, channels_mpsc, channels_spsc, arc_mutex, send_sync, rayon_parallelism, error_handling_result, anyhow_thiserror, unsafe_rust, ffi, procedural_macros

### Python (35)
generators, decorators, context_managers, descriptors, metaclasses, asyncio, event_loop, coroutines, tasks_futures, async_comprehensions, typing_module, generics_pep695, protocols, dataclasses, attrs, pydantic, slots_memory, weakrefs, gc_module, interning, import_system, namespace_packages, relative_imports, dunder_methods, iteration_protocols, slicing_advanced, walrus_operator, pattern_matching, exception_chaining, contextvars, concurrent_futures, multiprocessing, threading_gil, ctypes_cffi, c_extensions

### Mathematics (40)
linear_equations, quadratic_equations, polynomial_theory, inequalities, exponents_logarithms, sequences_series, complex_numbers, set_theory, relations_functions, limits, continuity, differentiation, chain_rule, optimization, taylor_series, integration_techniques, definite_integrals, improper_integrals, multivariable_calculus, partial_derivatives, lagrange_multipliers, vector_calculus, line_integrals, greens_theorem, vectors, matrices, determinants, eigenvalues_eigenvectors, svd, vector_spaces, linear_transformations, inner_product_spaces, probability_foundations, conditional_probability, bayes_theorem, random_variables, distributions_discrete, distributions_continuous, descriptive_statistics, hypothesis_testing

## Usage

### Load with Hugging Face `datasets`

```python
from datasets import load_dataset

# Load a single subtopic
ds = load_dataset("json", data_files="data/rust__ownership.jsonl", split="train")

# Load all files in a domain
import glob
ds = load_dataset("json", data_files=glob.glob("data/math__*.jsonl"), split="train")
```

### Load with raw Python

```python
import json

with open("data/python__asyncio.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        user_msg = entry["messages"][0]["content"]
        assistant_msg = entry["messages"][1]["content"]
        # ... use for training
```

### Use with `transformers` SFTTrainer

```python
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

ds = load_dataset("json", data_files="data/*.jsonl", split="train")
trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    args=SFTConfig(output_dir="./sft-out"),
)
trainer.train()
```

## Reproducing the dataset

```bash
git clone https://github.com/AFKmoney/neuro-code-math-dataset.git
cd neuro-code-math-dataset
python3 scripts/generate.py        # generates all 150 files in ~60 seconds
python3 scripts/build_index.py     # rebuilds data/index.jsonl
```

The generator is deterministic (seeded with `random.seed(42)` and per-idx permutation seeds), so the same commit will always produce the same dataset byte-for-byte.

Override the per-file entry count:

```bash
ENTRIES_PER_FILE=1000 python3 scripts/generate.py    # quick sample run
ENTRIES_PER_FILE=12000 python3 scripts/generate.py   # larger run (~1.8M entries)
```

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@misc{neuro-code-math-dataset,
  title  = {neuro-code-math-dataset: A 1B-token JSONL training dataset across neuroscience, Rust, Python, and mathematics},
  author = {AFKmoney},
  url    = {https://github.com/AFKmoney/neuro-code-math-dataset},
  year   = {2026},
}
```
