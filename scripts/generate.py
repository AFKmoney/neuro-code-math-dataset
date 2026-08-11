#!/usr/bin/env python3
"""
neuro-code-math-dataset generator
=================================
Generates a massive JSONL training dataset for LLM fine-tuning across 4 domains:

  1. APPLIED NEUROSCIENCE  (40 subtopics)  — real research areas
  2. RUST                   (35 subtopics)  — ownership, traits, async, etc.
  3. PYTHON                 (35 subtopics)  — generators, decorators, asyncio, etc.
  4. MATH                   (40 subtopics)  — algebra, calculus, linear algebra,
                                              probability, statistics, discrete, etc.

Layout: 4 domains x 150 subtopics = 150 JSONL files.
Each file contains ~7,500 chat-formatted entries.
Total target: ~1,140,000 entries, ~1B tokens.

All content is in English and instruction-tuning oriented
(chat messages format: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}).

Architecture for variety:
  - 15 prompt templates per domain
  - 10 audiences x 10 perspectives x 10 formats = 1000 variation tuples
  - Per subtopic, each entry gets a unique (template_idx, audience, perspective, format) tuple
  - Dedup by user-content hash
  - Long, structured assistant responses (~500-900 tokens each)

Usage:
    python3 generate.py                # full run (~7,500 entries/file)
    ENTRIES_PER_FILE=100 python3 generate.py    # quick sample run
"""

import json
import os
import sys
import random
import hashlib
import time
from multiprocessing import Pool, cpu_count

random.seed(42)

OUT_DIR = "/home/z/my-project/neuro-code-math/data"
os.makedirs(OUT_DIR, exist_ok=True)

# Default: ~7,500 entries/file × 150 files = ~1.125M entries (target ~1B tokens)
ENTRIES_PER_FILE = int(os.environ.get("ENTRIES_PER_FILE", "7500"))

# ---------------------------------------------------------------------------
# 150 SUBTOPICS — 4 domains
# ---------------------------------------------------------------------------
SUBTOPICS = {
    # =====================================================================
    # 1. APPLIED NEUROSCIENCE — 40 subtopics
    # =====================================================================
    "applied_neuroscience": [
        "neural_coding", "neural_decoding", "population_coding", "temporal_coding",
        "rate_coding", "sparse_coding", "predictive_coding", "bayesian_brain",
        "free_energy_principle", "active_inference", "global_workspace_theory",
        "integrated_information_theory", "higher_order_theories",
        "predictive_processing", "hierarchical_processing", "cortical_minicolumns",
        "thalamic_reticular_nucleus", "basal_ganglia_circuits", "cerebellar_computation",
        "hippocampal_formation", "entorhinal_grid_cells", "place_cells",
        "head_direction_cells", "theta_oscillations", "gamma_oscillations",
        "cross_frequency_coupling", "sharp_wave_ripples", "sleep_replay",
        "consolidation", "long_term_potentiation", "long_term_depression",
        "spike_timing_dependent_plasticity", "homeostatic_plasticity",
        "metaplasticity", "dendritic_computation", "axonal_computation",
        "neuromodulation", "dopamine_reward", "serotonin_modulation",
        "neural_manifolds",
    ],
    # =====================================================================
    # 2. RUST — 35 subtopics
    # =====================================================================
    "rust": [
        "ownership", "borrowing", "lifetimes", "mutable_references",
        "slice_types", "strings_str_vs_string", "traits", "trait_objects",
        "generic_types", "where_clauses", "associated_types", "default_impls",
        "trait_bounds", "newtype_pattern", "phantom_types", "drop_trait",
        "deref_trait", "from_into_traits", "iterators", "iterator_adapters",
        "collect_into", "async_await", "futures", "pinning",
        "tokio_runtime", "channels_mpsc", "channels_spsc", "arc_mutex",
        "send_sync", "rayon_parallelism", "error_handling_result",
        "anyhow_thiserror", "unsafe_rust", "ffi", "procedural_macros",
    ],
    # =====================================================================
    # 3. PYTHON — 35 subtopics
    # =====================================================================
    "python": [
        "generators", "decorators", "context_managers", "descriptors",
        "metaclasses", "asyncio", "event_loop", "coroutines",
        "tasks_futures", "async_comprehensions", "typing_module", "generics_pep695",
        "protocols", "dataclasses", "attrs", "pydantic",
        "slots_memory", "weakrefs", "gc_module", "interning",
        "import_system", "namespace_packages", "relative_imports", "dunder_methods",
        "iteration_protocols", "slicing_advanced", "walrus_operator", "pattern_matching",
        "exception_chaining", "contextvars", "concurrent_futures", "multiprocessing",
        "threading_gil", "ctypes_cffi", "c_extensions",
    ],
    # =====================================================================
    # 4. MATH — 40 subtopics
    # =====================================================================
    "math": [
        "linear_equations", "quadratic_equations", "polynomial_theory",
        "inequalities", "exponents_logarithms", "sequences_series",
        "complex_numbers", "set_theory", "relations_functions",
        "limits", "continuity", "differentiation",
        "chain_rule", "optimization", "taylor_series",
        "integration_techniques", "definite_integrals", "improper_integrals",
        "multivariable_calculus", "partial_derivatives", "lagrange_multipliers",
        "vector_calculus", "line_integrals", "greens_theorem",
        "vectors", "matrices", "determinants",
        "eigenvalues_eigenvectors", "svd", "vector_spaces",
        "linear_transformations", "inner_product_spaces",
        "probability_foundations", "conditional_probability", "bayes_theorem",
        "random_variables", "distributions_discrete", "distributions_continuous",
        "descriptive_statistics", "hypothesis_testing",
    ],
}

# Sanity check counts
assert len(SUBTOPICS["applied_neuroscience"]) == 40, "neuro must be 40"
assert len(SUBTOPICS["rust"]) == 35, "rust must be 35"
assert len(SUBTOPICS["python"]) == 35, "python must be 35"
assert len(SUBTOPICS["math"]) == 40, "math must be 40"
TOTAL_FILES = sum(len(v) for v in SUBTOPICS.values())
assert TOTAL_FILES == 150, f"expected 150 files, got {TOTAL_FILES}"

# ---------------------------------------------------------------------------
# VARIATION MATRICES — 10 × 10 × 10 = 1000 tuples per subtopic
# ---------------------------------------------------------------------------
AUDIENCES = [
    "a beginner who has never seen the concept before",
    "an intermediate practitioner looking to deepen understanding",
    "an advanced expert who wants a rigorous treatment",
    "a student preparing for an exam",
    "a professional applying the concept at work",
    "a teacher preparing a lesson plan",
    "a self-learner studying independently",
    "a manager who needs a high-level overview",
    "a researcher investigating the underlying theory",
    "a reviewer evaluating the quality of the reasoning",
]

PERSPECTIVES = [
    "from a theoretical foundations angle",
    "from a practical application angle",
    "from a historical development angle",
    "from a comparative perspective (contrasting alternatives)",
    "from a critical/limitation-focused perspective",
    "from a teach-the-concept angle",
    "from an everyday-life angle",
    "from a professional/workplace angle",
    "from a debugging/error-correction angle",
    "from an optimization/mastery angle",
]

FORMATS = [
    "step-by-step procedure",
    "bulleted checklist",
    "worked example with concrete numbers",
    "scenario walkthrough",
    "Q&A dialogue",
    "compare-and-contrast table description",
    "decision tree narrative",
    "common mistakes and corrections",
    "before/after transformation",
    "principles-then-application",
]


def _msg(role, content):
    return {"role": role, "content": content}


def _wrap(user, assistant):
    return {"messages": [_msg("user", user), _msg("assistant", assistant)]}


# ---------------------------------------------------------------------------
# SUBTOPIC SEED DEFINITIONS — rich content for every subtopic
# Each entry: { "def": str, "principles": [5], "examples": [3], "mistakes": [5] }
# ---------------------------------------------------------------------------
SEEDS = {}

# === APPLIED NEUROSCIENCE SEEDS (40) ===========================
NEURO_SEEDS = {
    "neural_coding": {
        "def": "the study of how the nervous system represents, transmits, and transforms information in patterns of neural activity such as spike trains, firing rates, and population states",
        "principles": ["Information is carried in both individual spike timing and population firing rates", "Multiple codes can coexist within the same circuit", "Coding schemes are constrained by noise and metabolic cost", "Decoding requires a probabilistic model of the encoding", "Codes are often redundancy-reduced and adapted to natural statistics"],
        "examples": ["Orientation columns in V1 encoding edge direction via firing rate", "Spike-timing codes in the auditory brainstem encoding microsecond interaural differences", "Population codes in motor cortex encoding intended reach direction"],
        "mistakes": ["Treating firing rate as the only code when timing also carries information", "Ignoring trial-to-trial variability when interpreting single-neuron responses", "Confusing correlation between spikes and stimulus with causal encoding", "Assuming a single code is used throughout the brain", "Overlooking the role of neuromodulators in reshaping codes"],
    },
    "neural_decoding": {
        "def": "the inverse problem of inferring the stimulus, behavior, or internal state from observed neural activity, typically using probabilistic estimators or learned readouts",
        "principles": ["Decoding accuracy upper-bounds what downstream circuits can extract", "Bayesian decoders incorporate prior distributions over stimuli", "Linear decoders often perform surprisingly well on population activity", "Decoding is sensitive to the bin size and filtering of the spike train", "Decoding can reveal what information is available even if the animal does not use it"],
        "examples": ["Reconstructing viewed images from fMRI voxels in early visual cortex", "Decoding intended arm movement from motor cortical ensemble activity for brain-machine interfaces", "Inferring the presented odor from olfactory bulb glomerular activation patterns"],
        "mistakes": ["Equating decodable information with behaviorally used information", "Using decoders trained on one stimulus distribution to test another without recalibration", "Ignoring correlations between neurons when constructing the decoder", "Reporting decoding accuracy without confidence intervals", "Confusing population-level decoding with single-cell representation"],
    },
    "population_coding": {
        "def": "a scheme in which information about a stimulus or motor plan is distributed across the joint activity of many neurons, often with overlapping tuning curves",
        "principles": ["Population codes tolerate single-cell noise via redundancy", "Information grows sublinearly with correlated neurons", "Center-of-mass and Bayesian readouts approximate optimal decoders", "Tuning curve width and correlation structure jointly determine coding precision", "Population activity often lies on low-dimensional manifolds"],
        "examples": ["Motor cortical neurons with cosine tuning to reach direction, decoded as a population vector", "MT neurons coding motion direction, pooled to discriminate near-threshold motion", "Hippocampal place cell ensembles jointly representing the animal's location"],
        "mistakes": ["Treating each neuron independently when computing population information", "Confusing the dimensionality of activity with the dimensionality of the code", "Ignoring trial-averaged versus single-trial distinctions", "Assuming Gaussian noise when distributions are heavy-tailed", "Forgetting that correlations can be informative, not just noise"],
    },
    "temporal_coding": {
        "def": "a coding scheme in which the precise timing of spikes relative to a reference (stimulus, oscillation phase, or other spikes) carries information beyond the average rate",
        "principles": ["Phase-locked spikes can encode features at sub-millisecond resolution", "Synchrony across neurons can signal stimulus salience", "Latency codes can convey information before rate codes stabilize", "Temporal codes require temporal reference frames to be decodable", "Temporal precision is bounded by membrane time constants and synaptic jitter"],
        "examples": ["Phase precession of hippocampal place cells relative to theta oscillations", "Auditory nerve spikes phase-locked to low-frequency sound waveforms", "Spike-timing codes in electric fish encoding sub-microsecond timing differences"],
        "mistakes": ["Claiming temporal coding from rate differences alone without timing evidence", "Ignoring the reference signal that gives timing meaning", "Confusing precise timing in vitro with operation in noisy in vivo conditions", "Treating any oscillation as evidence of temporal coding", "Overlooking biophysical limits on temporal precision"],
    },
    "rate_coding": {
        "def": "a coding scheme in which the average number of spikes fired over a time window conveys information about a stimulus or state",
        "principles": ["Rate codes average out noise but discard timing information", "Rate coding accuracy improves with integration window length", "Many sensory neurons show monotonic or tuned rate responses to stimulus parameters", "Rate codes are robust to synaptic jitter and conduction delays", "Metabolic cost scales with spike rate, favoring sparse rate codes"],
        "examples": ["Retinal ganglion cell firing rate increasing with light intensity", "Touch receptor firing rate scaling with pressure amplitude", "Motor cortical firing rate scaling with required force"],
        "mistakes": ["Equating rate coding with simplicity — it can coexist with temporal codes", "Assuming a fixed integration window across stimuli", "Treating trial-averaged tuning curves as the code used in single trials", "Ignoring adaptation that reshapes rate responses over time", "Conflating spikes per second with bits per second"],
    },
    "sparse_coding": {
        "def": "a representational scheme in which only a small fraction of available neurons are active at any moment, trading metabolic cost for representational capacity and interpretability",
        "principles": ["Sparse codes maximize representational capacity per active unit", "Sparse activity reduces metabolic cost", "Sparsity improves learned feature interpretability", "Overcomplete sparse dictionaries can represent natural statistics efficiently", "Sparsity and redundancy trade off against noise robustness"],
        "examples": ["V1 simple cells with sparse, localized receptive fields matching natural image patches", "Olfactory cortex neurons sparsely activated by odor identity", "Dentate gyrus granule cells generating sparse, decorrelated representations for pattern separation"],
        "mistakes": ["Confusing lifetime sparsity with population sparsity", "Assuming sparse codes must be binary", "Treating sparsity as an end rather than a means", "Forgetting that sparsity in firing requires sufficient population size for coverage", "Equating sparse activations with sparse connectivity"],
    },
    "predictive_coding": {
        "def": "a theoretical framework in which the brain continually generates predictions about incoming sensory signals and propagates prediction errors upward, updating internal models to minimize surprise",
        "principles": ["Only prediction errors need to be transmitted up the hierarchy", "Feedback carries predictions; feedforward carries residuals", "Top-down priors can suppress or sharpen bottom-up signals", "Prediction error minimization is equivalent to variational free energy minimization", "Attention can be framed as optimizing the precision-weighting of errors"],
        "examples": ["V1 responses suppressed by predictable visual textures (repetition suppression)", "Mismatch negativity EEG signals for unexpected auditory deviants", "Direction-selective neurons predicting moving object position under occlusion"],
        "mistakes": ["Treating predictive coding as a literal claim that every cortex does Bayes", "Confusing prediction errors with raw sensory residuals", "Assuming prediction and error units map 1:1 to known cell types without evidence", "Forgetting that predictions need their own generative model", "Equating suppression of predictable stimuli with evidence for predictive coding alone"],
    },
    "bayesian_brain": {
        "def": "the hypothesis that the brain represents and combines sensory information with prior beliefs according to Bayesian probability to compute near-optimal posterior estimates",
        "principles": ["Priors shape perception toward more probable interpretations", "Likelihoods reflect sensory uncertainty; wider distributions mean more noise", "Combining priors and likelihoods via Bayes' rule yields posteriors", "Optimal cue combination weights modalities by their inverse variance", "Suboptimalities can reveal computational or representational constraints"],
        "examples": ["Speed perception biased toward prior of slow motion", "Visual-haptic integration following minimum-variance predictions", "Interval reproduction biased toward the mean of the prior distribution"],
        "mistakes": ["Assuming Bayesian optimality implies explicit probability calculations", "Equating behavior consistent with Bayes with neural implementation of Bayes", "Forgetting that priors must themselves be learned", "Treating suboptimal behavior as evidence against Bayesian brains", "Confusing optimal cue combination with optimal decision-making"],
    },
    "free_energy_principle": {
        "def": "a mathematical framework stating that any self-organizing system at non-equilibrium steady state must minimize its variational free energy, equivalently minimizing surprise and maintaining its structural integrity",
        "principles": ["Free energy bounds surprise (negative log evidence) of observations", "Minimizing free energy requires either changing predictions or changing the world via action", "Perception and action both serve free-energy minimization", "Active inference generalizes optimal control and reinforcement learning", "Markov blankets formally separate internal states from external states"],
        "mistakes": ["Treating free energy as a metaphor rather than a variational bound", "Conflating variational free energy with thermodynamic free energy", "Assuming the principle makes specific neural predictions without auxiliary assumptions", "Overextending the framework to systems where Markov blankets are ill-defined", "Confusing free-energy minimization with simple error minimization"],
        "examples": ["Homeostatic regulation as active inference over interoceptive signals", "Saccades sampling the world to reduce visual surprise", "Action selection as inference under the free-energy bound"],
    },
    "active_inference": {
        "def": "a theory casting action and perception as two sides of minimizing expected free energy, where agents act to confirm predictions and reduce uncertainty about the world",
        "principles": ["Action and perception jointly minimize expected free energy", "Epistemic actions reduce uncertainty; pragmatic actions achieve preferred states", "Policy selection follows the principle of precision-weighted expected free energy", "Habit learning corresponds to reducing expected free energy of habitual policies", "Active inference unifies exploration, exploitation, and homeostasis"],
        "examples": ["Saccading to informative locations under high uncertainty", "Reaching to grasp an object to confirm its predicted shape", "Thermoregulatory behaviors driven by interoceptive prediction errors"],
        "mistakes": ["Equating active inference with reinforcement learning by another name", "Confusing expected free energy with immediate free energy", "Assuming active inference requires explicit planning in all systems", "Forgetting that preferences must be encoded as prior beliefs", "Treating any goal-directed behavior as active inference"],
    },
    "global_workspace_theory": {
        "def": "a theory proposing that conscious access occurs when information is broadcast across a distributed global workspace of cortical modules, making it available to many processes including report and planning",
        "principles": ["Information in the workspace is broadcast and globally accessible", "Workspace access is capacity-limited and serial in nature", "Unconscious processing is parallel and high-capacity", "Conscious access involves late, distributed cortical ignition", "Workspace contents can be reported, acted on, and rehearsed"],
        "examples": ["Masked stimuli failing to ignite the global workspace and remaining unconscious", "Sudden insight generating widespread cortical activation when a solution enters awareness", "Working memory maintenance of items in the global workspace"],
        "mistakes": ["Treating global workspace as a literal anatomical structure", "Equating workspace access with attention", "Confusing reportability with consciousness in locked-in patients", "Assuming workspace theory fully specifies the neural mechanism", "Forgetting that workspace access is graded, not all-or-none"],
    },
    "integrated_information_theory": {
        "def": "a theory that identifies consciousness with the integrated information (Phi) generated by a system above and beyond its parts, treating consciousness as a fundamental property of certain causal structures",
        "principles": ["Phi measures cause-effect structure irreducibility to parts", "Consciousness corresponds to maxima of integrated information", "Quality of experience corresponds to the geometry of cause-effect repertoires", "IIT is a phenomenological-first theory, deriving axioms from experience", "Reductionism is rejected — the whole is genuinely more than parts"],
        "examples": ["Computing Phi for simple logic gates to compare conscious potential", "Distinguishing cortical architectures from feedforward nets by their Phi", "Predicting that cerebellar circuits, despite many neurons, have low Phi and low consciousness"],
        "mistakes": ["Treating Phi as a practical measure for real brains — it is computationally intractable", "Equating high Phi with consciousness without checking the axioms", "Confusing IIT's ontological claims with operational measures", "Assuming IIT is testable in its strong form with current technology", "Reading IIT as a neuroscientific theory rather than a phenomenological-axiomatic one"],
    },
    "higher_order_theories": {
        "def": "theories of consciousness proposing that a mental state is conscious when represented by a higher-order representation, such as a thought or perception about that state",
        "principles": ["Consciousness requires meta-representation of first-order content", "Higher-order representations can be introspective or non-introspective", "Animals lacking higher-order machinery may have unconscious first-order states", "Lesions to higher-order areas can abolish conscious report", "Higher-order theories are representationalist at their core"],
        "examples": ["Prefrontal cortex representing visual content in ventral stream as conscious", "Split-brain patients showing dissociable higher-order access", "Blindsight as intact first-order processing without higher-order representation"],
        "mistakes": ["Equating higher-order theory with prefrontal cortex theory", "Confusing higher-order representation with attention", "Treating all higher-order theories as identical", "Assuming higher-order representations must be conscious themselves", "Forgetting that higher-order theories make different predictions about animals"],
    },
    "predictive_processing": {
        "def": "an umbrella framework for cortical computation in which hierarchical generative models continually predict sensory input, propagate prediction errors, and update beliefs to minimize surprise",
        "principles": ["Hierarchical generative models capture the causal structure of the world", "Predictions flow downward; errors flow upward", "Precision-weighting modulates the influence of errors", "Cortical columns implement predictive coding computations", "Learning updates priors and likelihoods to reduce long-term surprise"],
        "examples": ["V1 responses attenuated to predictable visual input", "Auditory cortex generating predictions about upcoming tones in music", "Cross-modal predictions from visual cortex shaping auditory responses"],
        "mistakes": ["Treating predictive processing as a single, well-specified theory", "Equating it with Bayesian brain without distinguishing commitments", "Assuming specific cortical cell types implement specific computations without evidence", "Confusing predictive processing with supervised learning", "Forgetting that predictive processing models require explicit generative models"],
    },
    "hierarchical_processing": {
        "def": "the organization of neural computation in cascaded stages where each level extracts increasingly abstract features from the previous level's output",
        "principles": ["Lower levels extract simple features; higher levels compose them", "Receptive field size grows with hierarchy level", "Invariance to nuisance variation increases up the hierarchy", "Feedback connections enable top-down modulation", "Hierarchical depth correlates with representational abstraction"],
        "examples": ["V1 edges → V2 textures → V4 shapes → IT object categories", "A1 tones → belt phonemes → STG words → STS sentences", "S1 touches → S2 shapes → insula body states"],
        "mistakes": ["Treating hierarchy as strictly feedforward", "Assuming each level has a single canonical computation", "Confusing anatomical hierarchy with functional hierarchy", "Forgetting that hierarchy is shared across modalities with cross-talk", "Equating deeper layers with better representations without task context"],
    },
    "cortical_minicolumns": {
        "def": "vertical columns of approximately 80-100 neurons spanning the cortical depth, hypothesized to be a fundamental computational unit of cerebral cortex",
        "principles": ["Minicolumns share feedforward input via thalamic projections", "Pyramidal cells within a column project to common targets", "Inter-columnar inhibition sharpens columnar response selectivity", "Columns may implement canonical computations such as normalization", "Columnar organization is visible across most mammalian cortex"],
        "examples": ["Orientation columns in V1 with shared orientation preference", "Ocular dominance columns separating input from the two eyes", "Barrel columns in rodent S1 corresponding to individual whiskers"],
        "mistakes": ["Treating minicolumns as discrete, isolated units", "Confusing minicolumns with macrocolumns or hypercolumns", "Assuming columnar organization implies columnar function", "Forgetting that columnar inputs include substantial horizontal connections", "Equating structural columns with the computational unit of cortex"],
    },
    "thalamic_reticular_nucleus": {
        "def": "a thin sheet of GABAergic neurons surrounding the thalamus that inhibits thalamic relay nuclei and gates the flow of information between thalamus and cortex",
        "principles": ["TRN neurons are entirely GABAergic and inhibitory", "TRN receives input from cortex and thalamus but does not project to cortex", "TRN controls thalamocortical rhythm generation including sleep spindles", "TRN is organized into sensory-modality-specific sectors", "TRN contributes to attentional gating of sensory input"],
        "examples": ["TRN generating sleep spindles during stage 2 NREM sleep", "TRN sectors corresponding to visual, auditory, and somatosensory pathways", "TRN lesion disrupting selective attention in animal models"],
        "mistakes": ["Treating TRN as a uniform structure", "Confusing TRN with other thalamic nuclei", "Assuming TRN is purely inhibitory in function — it can disinhibit via complex circuits", "Forgetting TRN's role in rhythm generation, not just gating", "Equating TRN activity with attention without controlling for arousal"],
    },
    "basal_ganglia_circuits": {
        "def": "a set of parallel cortico-basal ganglia-thalamo-cortical loops involved in action selection, reinforcement learning, habit formation, and motor scaling",
        "principles": ["Direct pathway facilitates selected actions; indirect pathway suppresses alternatives", "Dopaminergic input from SNc/VTA signals reward prediction error", "Loops are organized topically (motor, associative, limbic)", "Striatum is the main input structure receiving cortical excitation", "GPi/SNr output inhibits thalamic nuclei that gate cortical action"],
        "examples": ["Direct pathway activation promoting a reaching movement", "Indirect pathway suppressing competing motor programs", "Parkinsonian motor symptoms from dopaminergic cell loss in SNc"],
        "mistakes": ["Treating all basal ganglia loops as functionally equivalent", "Confusing direct/indirect pathways with go/no-go learning signals", "Forgetting the hyperdirect pathway through subthalamic nucleus", "Equating dopamine with hedonic pleasure rather than prediction error", "Assuming basal ganglia only handle motor control"],
    },
    "cerebellar_computation": {
        "def": "the signal processing performed by the cerebellum, characterized by sparse coding in granule cells, error-driven plasticity at parallel fiber-Purkinje cell synapses, and supervised learning of motor and cognitive predictions",
        "principles": ["Granule cell layer expands inputs into a high-dimensional sparse representation", "Purkinje cells integrate parallel fiber inputs and inhibit deep nuclei", "Climbing fibers from inferior olive provide teaching signals", "Long-term depression at parallel fiber-Purkinje synapses underlies motor learning", "Cerebellum implements forward models for state prediction"],
        "examples": ["Smooth pursuit eye movement gain adaptation", "Classical eyeblink conditioning with mossy fiber CS and climbing fiber US", "Saccadic accuracy calibration via cerebellar plasticity"],
        "mistakes": ["Treating cerebellum as purely motor — it also contributes to cognition", "Confusing LTD with LTP at parallel fiber-Purkinje synapses", "Assuming climbing fibers only signal errors — they also fire spontaneously", "Forgetting that deep cerebellar nuclei are the main output, not Purkinje cells", "Equating cerebellar learning with reinforcement learning — it is supervised"],
    },
    "hippocampal_formation": {
        "def": "a set of interconnected structures including the dentate gyrus, CA3, CA2, CA1, and subiculum, critical for episodic memory, spatial cognition, and relational binding",
        "principles": ["CA3 has recurrent collaterals supporting pattern completion", "Dentate gyrus performs pattern separation via sparse coding", "CA1 compares predictions from CA3 with sensory input from entorhinal cortex", "Sharp-wave ripples in CA1 replay sequences for consolidation", "Theta oscillations organize hippocampal processing during exploration"],
        "examples": ["CA3 pattern completion enabling memory recall from partial cues", "Dentate gyrus separating similar experiences into distinct memories", "CA1 place cells signaling location during spatial navigation"],
        "mistakes": ["Treating hippocampus as a single uniform structure", "Confusing episodic memory with semantic memory localization", "Assuming hippocampus only handles memory — it also contributes to navigation and imagination", "Forgetting that subfield-specific lesions have different effects", "Equating place cells with hippocampal function overall"],
    },
    "entorhinal_grid_cells": {
        "def": "neurons in medial entorhinal cortex that fire at multiple spatial locations arranged in a hexagonal triangular lattice, providing a metric for spatial navigation",
        "principles": ["Grid fields form hexagonal lattices with consistent spacing and orientation", "Grid spacing increases along the dorsolateral-to-ventromedial axis", "Grid cells likely implement path integration via attractor dynamics", "Grids align with environmental geometry and cues", "Grid cells coexist with head direction and border cells in MEC"],
        "examples": ["Hexagonal firing pattern in dorsal MEC during open-field exploration", "Grid realignment under environmental deformation", "Coordinated grid networks providing a population code for position"],
        "mistakes": ["Treating grid cells as the sole basis for spatial cognition", "Assuming grid spacing is fixed — it stretches in larger environments", "Confusing grid cells with place cells in downstream hippocampus", "Forgetting that grid cells fire in non-spatial contexts too", "Equating hexagonal firing with a perfect theoretical lattice"],
    },
    "place_cells": {
        "def": "pyramidal neurons in hippocampal CA1 and CA3 that fire selectively when an animal occupies a specific location in its environment, termed the cell's place field",
        "principles": ["Place fields remap between environments and across contexts", "Place cells can represent past and future locations during replay and preplay", "Place field stability depends on NMDA-dependent plasticity", "Place cells are modulated by theta oscillations and show theta phase precession", "Place cells encode more than location — they also reflect experience, reward, and context"],
        "examples": ["CA1 place field remapping when an animal enters a novel environment", "Backward replay of place cell sequences during sharp-wave ripples at reward", "Theta phase precession of place cells as a rat traverses a linear track"],
        "mistakes": ["Equating place cells with a GPS — they reflect cognitive maps, not coordinates", "Assuming place fields are stable across all conditions", "Confusing place cells with grid cells in entorhinal cortex", "Forgetting that place cells can fire outside their primary field", "Treating place cells as purely spatial — they also encode non-spatial variables"],
    },
    "head_direction_cells": {
        "def": "neurons that fire maximally when an animal's head is oriented in a specific direction in the horizontal plane, regardless of location, found in multiple brain regions including postsubiculum, anterior thalamus, and entorhinal cortex",
        "principles": ["Head direction cells have a single preferred direction independent of position", "HD system forms a ring attractor encoding directional heading", "HD signals depend on vestibular, visual, and proprioceptive inputs", "HD cells cohere across brain regions as a network", "HD signal updates with angular velocity to enable path integration"],
        "examples": ["Postsubicular HD cell firing when rat faces north regardless of location", "Anterior thalamic HD neurons maintaining directional tuning in darkness", "HD cell preferred direction shifting with environmental rotation"],
        "mistakes": ["Confusing head direction cells with place cells", "Assuming HD cells fire for eye direction rather than head direction", "Treating HD system as purely visual — vestibular inputs are essential", "Forgetting that HD cells exist in many brain regions, not one", "Equating HD tuning with allocentric compass direction"],
    },
    "theta_oscillations": {
        "def": "hippocampal and cortical oscillations in the 4-12 Hz range associated with exploration, memory encoding, navigation, and the coordination of neural assemblies",
        "principles": ["Theta is prominent during voluntary movement and REM sleep", "Theta organizes spike timing via phase precession", "Theta-gamma coupling supports working memory and sequence ordering", "Theta phase gates plasticity windows in hippocampus", "Medial septum is the primary pacemaker for hippocampal theta"],
        "examples": ["Hippocampal theta rhythm during rat exploration of a maze", "Theta phase precession of place cells as a rat traverses a field", "Theta power increasing with memory demand in humans"],
        "mistakes": ["Equating theta with a single function — it has many roles", "Confusing hippocampal theta with cortical theta recorded in EEG", "Assuming theta is always locally generated — it can be volume-conducted", "Treating theta frequency as fixed — it varies with behavior and species", "Forgetting that theta can both facilitate and gate plasticity depending on phase"],
    },
    "gamma_oscillations": {
        "def": "neural oscillations in the 30-120 Hz range associated with local cortical computation, perceptual binding, attention, and the coordination of feedforward and feedback signals",
        "principles": ["Gamma reflects the rhythm of local inhibitory-excitatory loops", "Gamma power correlates with attention and stimulus salience", "Low gamma (30-50 Hz) and high gamma (60-120 Hz) have distinct mechanisms", "Gamma-theta coupling organizes item-context associations", "Gamma synchrony can bind features across cortical regions"],
        "examples": ["Visual cortex gamma increasing with attended stimuli", "Hippocampal gamma coupling to theta during working memory tasks", "Auditory cortex gamma entrainment to rhythmic sounds"],
        "mistakes": ["Treating gamma as a single uniform phenomenon", "Confusing gamma with high-frequency artifact in EEG", "Assuming gamma synchrony always means binding", "Equating gamma power with cortical activation generally", "Forgetting that gamma can reflect both spiking and synaptic activity"],
    },
    "cross_frequency_coupling": {
        "def": "the interaction between oscillations at different frequencies, most commonly phase-amplitude coupling where the phase of a slow oscillation modulates the amplitude of a faster one",
        "principles": ["Theta-gamma coupling in hippocampus supports memory sequences", "Phase of slow oscillation gates amplitude windows of fast oscillations", "CFC can organize multi-scale computation across brain regions", "CFC metrics must account for non-stationarity and noise", "CFC may reflect canonical neural computations or epiphenomenal correlations"],
        "examples": ["Hippocampal theta phase modulating gamma amplitude during spatial memory", "Slow cortical oscillation phase gating spindle amplitude during sleep", "Alpha phase modulating visual gamma during attention"],
        "mistakes": ["Equating CFC with functional coupling — it may be correlational", "Using inappropriate metrics that confound CFC with non-sinusoidal oscillations", "Assuming CFC directionality without proper analysis", "Treating all CFC as theta-gamma — many other pairs exist", "Forgetting that CFC measures depend on signal-to-noise"],
    },
    "sharp_wave_ripples": {
        "def": "transient high-frequency oscillatory events (140-200 Hz) in hippocampal CA1 during quiet wakefulness and slow-wave sleep, associated with memory replay and consolidation",
        "principles": ["SWRs are generated by CA3 excitation of CA1", "SWRs carry sequential replay of past experiences", "SWR disruption impairs spatial memory consolidation", "SWR content can reflect past, future, or imagined trajectories", "SWRs are synchronized across hippocampal-cortical networks"],
        "examples": ["CA1 sharp-wave ripples during rat rest after maze exploration", "Replay of place cell sequences during SWRs matching prior behavior", "Pre-play of future trajectories during SWRs before choice"],
        "mistakes": ["Treating SWRs as purely rest phenomena — they occur during exploration too", "Confusing ripples with gamma oscillations", "Assuming all SWRs carry replay — many are noise-like", "Equating SWR disruption with memory impairment specifically", "Forgetting that SWRs coordinate with cortical slow oscillations during sleep"],
    },
    "sleep_replay": {
        "def": "the reactivation during sleep of neural sequence patterns experienced during prior wakefulness, primarily mediated by hippocampal sharp-wave ripples and coordinated with neocortical slow oscillations",
        "principles": ["Replay occurs during NREM sleep, especially slow-wave and stage 2", "Replay is compressed relative to the original experience", "Replay supports memory consolidation by transferring hippocampal to neocortical storage", "Replay can be forward or reverse relative to experience", "Replay content is biased toward reward-related experiences"],
        "examples": ["Hippocampal place cell sequences replaying a prior maze trajectory during sleep", "Cortical-hippocampal coupling during slow oscillation up-states coordinating replay", "Reverse replay of place cell sequences at reward sites"],
        "mistakes": ["Treating all reactivation during sleep as replay — it includes preplay too", "Assuming replay is veridical — it can be compressed, skipped, or recombined", "Equating replay with dreaming", "Confusing sleep replay with waking replay", "Forgetting that replay is selective — only some experiences reactivate"],
    },
    "consolidation": {
        "def": "the process by which memories become stable and resistant to disruption over time, involving systems-level reorganization from hippocampal to neocortical dependence",
        "principles": ["Consolidation unfolds over hours to years", "Systems consolidation transfers memory dependence from hippocampus to neocortex", "Synaptic consolidation stabilizes individual synapses over minutes to hours", "Reconsolidation can re-open consolidated memories after retrieval", "Sleep, especially slow-wave sleep, facilitates consolidation"],
        "examples": ["Hippocampal lesions causing retrograde amnesia for recent but not remote memories", "Sleep after learning improving declarative memory retention", "Reconsolidation allowing memory updating after recall"],
        "mistakes": ["Treating consolidation as a single process — synaptic and systems consolidation differ", "Assuming consolidation is one-shot — memories can reconsolidate", "Confusing consolidation with simple forgetting", "Equating systems consolidation with hippocampal inactivation", "Forgetting that emotional salience modulates consolidation via amygdala"],
    },
    "long_term_potentiation": {
        "def": "a persistent strengthening of synapses based on recent patterns of activity, widely considered a cellular mechanism underlying learning and memory",
        "principles": ["LTP requires coincident pre- and post-synaptic activity", "NMDA receptor-dependent LTP requires calcium influx through NMDA receptors", "AMPA receptor trafficking into the postsynaptic density expresses LTP", "LTP has early and late phases distinguished by protein synthesis dependence", "LTP is input-specific and associative"],
        "examples": ["NMDA-dependent LTP at Schaffer collateral-CA1 synapses", "Theta-burst stimulation inducing LTP in hippocampal slices", "Visual cortex LTP during critical period for ocular dominance plasticity"],
        "mistakes": ["Treating LTP as the only plasticity mechanism — LTD matters too", "Assuming LTP always means memory — it can be homeostatic or pathological", "Confusing LTP induction with LTP expression", "Equating NMDA-dependent LTP with all forms of LTP", "Forgetting that LTP is reversible and can be depotentiated"],
    },
    "long_term_depression": {
        "def": "a persistent weakening of synapses following specific patterns of low-frequency activity, complementing LTP as a bidirectional plasticity mechanism",
        "principles": ["LTD can be NMDA-dependent or metabotropic glutamate receptor-dependent", "LTD requires modest postsynaptic calcium elevation, lower than LTP threshold", "LTD involves AMPA receptor removal from the postsynaptic density", "LTD is critical for synaptic competition and forgetting", "Cerebellar LTD at parallel fiber-Purkinje synapses underlies motor learning"],
        "examples": ["NMDA-dependent LTD at CA1 synapses with low-frequency stimulation", "Cerebellar LTD pairing parallel fiber and climbing fiber activity", "mGluR-dependent LTD in hippocampus required for some forms of learning"],
        "mistakes": ["Treating LTD as simply 'anti-LTP' — it has distinct molecular mechanisms", "Assuming LTD always means forgetting — it can support learning", "Confusing NMDA-dependent and mGluR-dependent LTD", "Equating synaptic depression with neuronal depression", "Forgetting that LTD is bidirectional and can be de-depressed"],
    },
    "spike_timing_dependent_plasticity": {
        "def": "a form of Hebbian plasticity in which the sign and magnitude of synaptic change depend on the relative timing of pre- and post-synaptic spikes, with millisecond precision",
        "principles": ["Pre-before-post within ~20 ms causes LTP", "Post-before-pre within ~20 ms causes LTD", "STDP windows are asymmetric and decay exponentially", "STDP can implement competitive learning and self-organization", "STDP requires backpropagating action potentials in dendrites"],
        "examples": ["STDP in hippocampal cultures pairing pre- and post-synaptic spikes", "STDP in visual cortex shaping orientation selectivity during development", "STDP at inhibitory synapses with reversed timing requirements"],
        "mistakes": ["Treating STDP as the only plasticity rule — many synapses use rate-based rules", "Assuming STDP windows are fixed — they depend on neuromodulators and location", "Confusing pairing-induced STDP with in vivo plasticity", "Equating STDP with Hebbian learning without considering the timing requirement", "Forgetting that STDP interacts with homeostatic mechanisms"],
    },
    "homeostatic_plasticity": {
        "def": "a set of plasticity mechanisms that stabilize neural activity within target ranges over hours to days, including synaptic scaling and intrinsic excitability adjustments",
        "principles": ["Synaptic scaling multiplicatively adjusts all synapses to maintain target firing", "Intrinsic plasticity adjusts ion channel densities to tune excitability", "Homeostatic plasticity operates on slower timescales than Hebbian plasticity", "Sleep may be a privileged time for homeostatic renormalization", "Homeostatic mechanisms prevent runaway excitation or silence"],
        "examples": ["Synaptic up-scaling after chronic activity blockade in cultured neurons", "Intrinsic excitability increases after prolonged silencing", "Sleep down-scaling of synapses strengthened during wakefulness (SHY hypothesis)"],
        "mistakes": ["Treating homeostatic plasticity as uniformly slow — some forms are rapid", "Assuming synaptic scaling is global — it can be compartment-specific", "Confusing homeostatic with Hebbian plasticity — they operate in parallel", "Equating homeostatic plasticity with sleep only — it occurs in wake too", "Forgetting that homeostatic setpoints themselves can be plastic"],
    },
    "metaplasticity": {
        "def": "the plasticity of plasticity, in which prior neural activity alters the rules governing future synaptic or intrinsic plasticity, often via changes in plasticity thresholds",
        "principles": ["Prior activity can raise the LTP threshold and lower the LTD threshold (BCM theory)", "Metaplasticity stabilizes learning over long timescales", "Metaplasticity can be homosynaptic or heterosynaptic", "Molecular substrates include NMDA receptor subunit composition changes", "Metaplasticity enables learning-rate adaptation"],
        "examples": ["Visual deprivation shifting the BCM modification threshold", "Prior LTP making subsequent LTP harder to induce", "Synaptic priming facilitating later plasticity"],
        "mistakes": ["Treating metaplasticity as a single mechanism — it has many forms", "Confusing metaplasticity with homeostatic plasticity — they overlap but differ", "Assuming metaplasticity always reduces plasticity — it can also facilitate", "Equating metaplasticity with synaptic tagging — they are distinct", "Forgetting that metaplasticity can be intrinsic, not just synaptic"],
    },
    "dendritic_computation": {
        "def": "the nonlinear signal processing performed by dendrites, including local spike generation, compartmentalized integration, and feature selectivity emerging from dendritic geometry and active conductances",
        "principles": ["Dendrites are not passive cables — they have voltage-gated channels", "Local dendritic spikes can occur independently of soma spikes", "Dendritic compartments isolate and combine inputs nonlinearly", "Dendritic inhibition can be localized and targeted", "Dendritic computation greatly expands single-neuron processing capacity"],
        "examples": ["CA1 pyramidal apical dendrites generating calcium spikes", "Layer 5 pyramidal basal dendrites producing NMDA spikes in response to clustered input", "Cerebellar Purkinje dendrites computing complex patterns via dendritic calcium spikes"],
        "mistakes": ["Treating dendrites as passive integrators", "Assuming somatic spikes are the only output", "Confusing dendritic spikes with somatic spikes", "Equating dendritic computation with dendritic plasticity — they are related but distinct", "Forgetting that dendritic morphology constrains computation"],
    },
    "axonal_computation": {
        "def": "the signal processing performed in axons beyond simple impulse conduction, including axonal initiation sites, branch point failures, and activity-dependent conduction velocity changes",
        "principles": ["Axon initial segment is the usual spike initiation site with high channel density", "Branch points can fail or amplify spikes based on geometry", "Myelination enables saltatory conduction and increases speed", "Axons can express voltage-gated channels along their length, enabling nonlinear processing", "Activity can alter axonal conduction velocity over multiple timescales"],
        "examples": ["AIS position tuning excitability in auditory neurons", "Branch point failures filtering spike propagation in hippocampal axons", "Demyelination disrupting saltatory conduction in multiple sclerosis"],
        "mistakes": ["Treating axons as simple output cables", "Assuming all spikes initiated at the soma propagate faithfully", "Confusing myelination with simple insulation — it involves channel clustering at nodes", "Equating axonal conduction velocity with a fixed property", "Forgetting that axons can have local protein synthesis supporting plasticity"],
    },
    "neuromodulation": {
        "def": "the diffuse regulation of neural circuit properties by neurotransmitters such as dopamine, serotonin, norepinephrine, and acetylcholine, acting over slower timescales and broader spatial scales than fast synaptic transmission",
        "principles": ["Neuromodulators act via GPCRs with seconds-to-minutes effects", "Neuromodulators can alter excitability, plasticity, and signal-to-noise", "Single neuromodulators can have opposing effects in different circuits via distinct receptor subtypes", "Neuromodulatory nuclei project widely throughout the brain", "Neuromodulation is central to arousal, attention, mood, and learning"],
        "examples": ["Dopamine from VTA modulating striatal plasticity and reward learning", "Acetylcholine from basal forebrain enhancing cortical attention", "Norepinephrine from locus coeruleus regulating arousal and vigilance"],
        "mistakes": ["Treating each neuromodulator as having a single function", "Equating dopamine with pleasure rather than prediction error or motivation", "Confusing neuromodulation with fast neurotransmission", "Assuming neuromodulators act uniformly across brain regions", "Forgetting that receptor subtypes mediate divergent effects"],
    },
    "dopamine_reward": {
        "def": "the dopaminergic signaling of reward prediction errors in which phasic dopamine responses encode the difference between expected and obtained reward, driving reinforcement learning",
        "principles": ["Phasic dopamine signals reward prediction error (RPE)", "Positive RPE potentiates actions; negative RPE depresses them", "Dopamine reinforces actions via D1 receptor activation in direct pathway", "Dopamine response transfers from reward to predictive cues over learning", "Tonic dopamine levels encode average reward rate and motivate effort"],
        "examples": ["VTA dopamine neuron firing to unexpected juice reward", "Dopamine response shifting to a conditioned stimulus after pairing with reward", "Negative RPE dips in dopamine when expected reward is omitted"],
        "mistakes": ["Equating dopamine with pleasure — it signals prediction, not hedonia", "Treating dopamine as monolithic — tonic and phasic signals differ", "Assuming dopamine always reinforces — it can also depress via D2 receptors", "Confusing RPE with salience or attention signals", "Forgetting that dopamine has roles beyond reward, including movement"],
    },
    "serotonin_modulation": {
        "def": "the serotonergic regulation of mood, impulsivity, appetite, and social behavior originating from the raphe nuclei, with complex effects on circuit dynamics via multiple receptor subtypes",
        "principles": ["Serotonin originates in dorsal and median raphe nuclei", "Serotonin acts on at least 14 receptor subtypes with diverse effects", "Serotonin generally opposes dopamine in reward and behavioral activation", "Serotonin contributes to impulse control and patience", "SSRIs elevate synaptic serotonin over weeks to treat depression"],
        "examples": ["Serotonin depletion increasing impulsivity in animal models", "Dorsal raphe serotonin neurons firing during waiting behavior", "SSRI treatment reducing depressive symptoms over weeks"],
        "mistakes": ["Equating serotonin with happiness — its role is more nuanced", "Treating serotonin as a single function — it has many roles", "Confusing acute serotonin effects with chronic SSRI mechanisms", "Assuming serotonin only affects mood — it also affects gut, platelets, and bone", "Forgetting that receptor subtypes mediate opposing effects"],
    },
    "neural_manifolds": {
        "def": "low-dimensional geometric structures embedded in high-dimensional neural population activity, capturing the relevant computational variables and revealing how circuits organize information",
        "principles": ["Population activity often lies on low-dimensional manifolds", "Manifold dimensionality reflects the number of relevant computational variables", "Manifolds can be discovered via PCA, GPFA, t-SNE, or UMAP", "Manifold geometry relates to computational properties like separability and stability", "Different tasks may engage different manifolds in the same circuit"],
        "examples": ["Motor cortex population activity during reaching lying on a low-dimensional manifold", "Prefrontal cortex manifolds reorganizing across task epochs", "Hippocampal manifolds encoding position and context simultaneously"],
        "mistakes": ["Equating low dimensionality with simplicity — manifolds can be complex", "Confusing the manifold with the activity itself", "Assuming a single manifold per circuit — multiple may coexist", "Misinterpreting dimensionality reduction artifacts as biological structure", "Forgetting that manifold geometry depends on behavioral context"],
    },
}
SEEDS.update(NEURO_SEEDS)

# === RUST SEEDS (35) ===========================
RUST_SEEDS = {
    "ownership": {
        "def": "the rule that every value in Rust has exactly one owner whose lifetime bounds the value's lifetime, and transferring or borrowing ownership is tracked at compile time to guarantee memory safety without garbage collection",
        "principles": ["Each value has exactly one owner at a time", "When the owner goes out of scope, the value is dropped", "Assignment of non-Copy types moves ownership rather than copying", "Function calls take ownership unless the parameter borrows", "Ownership transfer is a zero-cost compile-time concept"],
        "examples": ["String assignment moving the heap buffer from one variable to another", "Passing a Vec into a function transferring ownership and ending the caller's access", "Returning a value from a function transferring ownership back to the caller"],
        "mistakes": ["Trying to use a variable after its value has been moved", "Confusing Copy types like i32 with non-Copy types like String", "Assuming clone is automatic — it must be explicit for non-Copy types", "Forgetting that function parameters take ownership unless they borrow", "Believing ownership is a runtime concept — it is purely compile-time"],
    },
    "borrowing": {
        "def": "the mechanism by which code accesses a value without taking ownership, using references that the borrow checker verifies always point to valid data",
        "principles": ["Either one mutable reference or many immutable references, but not both", "References must always point to valid data", "Borrow lifetimes cannot exceed the lifetime of the borrowed data", "Borrowing is a zero-cost abstraction", "Shared references are Copy; mutable references are not"],
        "examples": ["Passing &String to a function so the caller retains ownership", "Iterating over &Vec<T> to borrow each element", "Holding &mut buffer to push bytes into a Vec in place"],
        "mistakes": ["Mixing mutable and immutable borrows of the same data", "Holding a reference after the owner is dropped", "Confusing &T and &mut T semantics", "Forgetting that method calls implicitly borrow self", "Assuming borrowing is expensive — it has no runtime cost"],
    },
    "lifetimes": {
        "def": "compile-time annotations describing how long references must remain valid, used by the borrow checker to ensure no reference outlives the data it points to",
        "principles": ["Every reference has a lifetime, often elided by the compiler", "Lifetimes ensure references never dangle", "Multiple lifetime parameters express relationships between references", "'static lifetime means the reference is valid for the entire program", "Lifetime bounds on generic types constrain reference validity"],
        "examples": ["A function returning &str with lifetime tied to one of its arguments", "A struct holding a reference annotated with a lifetime parameter", "'static str literals having the longest possible lifetime"],
        "mistakes": ["Confusing 'static as a guarantee of immortality rather than program scope", "Trying to return a reference to a local variable", "Over-annotating lifetimes when elision would work", "Misunderstanding lifetime variance and how it affects subtyping", "Forgetting that lifetimes are erased at runtime"],
    },
    "mutable_references": {
        "def": "references that allow modifying the borrowed value, denoted &mut T, with the constraint that no other reference to the same data may exist simultaneously",
        "principles": ["Only one mutable reference to a value at a time", "Mutable references cannot coexist with immutable references", "Mutable references are created with &mut expr", "A mutable reference gives exclusive access to the value", "Mutable borrows last until the last use, not the end of scope (NLL)"],
        "examples": ["&mut vec to push items into a Vec in a function", "Reborrowing a &mut slice to split it into two halves", "Mutating a field through a mutable reference to a struct"],
        "mistakes": ["Trying to take a mutable reference to a non-mut variable", "Holding two mutable references to the same data", "Forgetting that reborrowing can extend the use of a mutable reference", "Assuming mutable references are Copy — they are not", "Confusing the end of a mutable borrow with end of scope"],
    },
    "slice_types": {
        "def": "fat-pointer references to a contiguous subsequence of a collection, such as &[T] for slices of arrays and vectors or &str for string slices, carrying a pointer and length",
        "principles": ["Slices are views into contiguous data without owning it", "Slice types are fat pointers with pointer and length", "Borrowing a Vec as &[T] is a cheap coercion", "String slices &str are slices of UTF-8 bytes", "Slices enable writing functions that work with any contiguous collection"],
        "examples": ["&[1, 2, 3, 4] split into two slices with split_at", "Passing &str to a function so it works with both String and string literals", "Slicing a Vec to pass a portion to a function without copying"],
        "mistakes": ["Treating slices as owning collections", "Confusing &str with String — the former borrows, the latter owns", "Indexing a slice out of bounds causing a panic", "Forgetting that slices must point to valid memory", "Assuming slices are zero-cost across function boundaries — they pass two words"],
    },
    "strings_str_vs_string": {
        "def": "the distinction between str (an unsized UTF-8 string slice, always borrowed via &str) and String (an owned, growable UTF-8 string backed by a heap buffer)",
        "principles": ["&str is a borrowed view into UTF-8 bytes", "String is an owned, growable UTF-8 buffer", "String derefs to &str, enabling coercion", "&str can point to string literals, slices, or String contents", "Both enforce UTF-8 validity at the type level"],
        "examples": ["Storing user input in String and passing &str to processing functions", "Returning String from a function that builds text and accepting &str in helpers", "Concatenating with format! producing a new String"],
        "mistakes": ["Confusing &str and String ownership semantics", "Forgetting that not all byte sequences are valid UTF-8", "Excessively cloning String when &str would do", "Treating String as immutable — it is mutable through &mut", "Assuming str can exist standalone — it needs to be behind a pointer"],
    },
    "traits": {
        "def": "Rust's mechanism for defining shared behavior via a set of method signatures that types can implement, enabling abstraction, polymorphism, and code reuse without inheritance",
        "principles": ["Traits declare method signatures types can implement", "Traits can have default method implementations", "Traits enable polymorphism via trait objects or generics", "Traits can require other traits as supertraits", "Types can implement any number of traits"],
        "examples": ["Implementing Display for a custom type to enable printing", "Defining a Summary trait with a summarize method that types can implement", "Using Into and From traits to enable type conversions"],
        "mistakes": ["Treating traits as classes — there is no inheritance of fields", "Implementing foreign traits on foreign types (orphan rule violation)", "Confusing trait objects with generics", "Forgetting that trait methods can be called only if the trait is in scope", "Overusing traits when simple functions would suffice"],
    },
    "trait_objects": {
        "def": "values of unknown concrete type known only by their trait, implemented via fat pointers containing a pointer to the data and a pointer to a vtable of trait methods, denoted dyn Trait",
        "principles": ["Trait objects enable runtime polymorphism", "Trait objects are fat pointers with data and vtable pointers", "Only object-safe traits can be made into trait objects", "Trait objects incur dynamic dispatch cost", "Box<dyn Trait> is a common pattern for owning trait objects"],
        "examples": ["Vec<Box<dyn Draw>> holding different shape types that can all draw", "Returning Box<dyn Error> from a function for flexible error handling", "Storing heterogeneous plugins as Vec<Box<dyn Plugin>>"],
        "mistakes": ["Trying to create trait objects of non-object-safe traits", "Confusing trait objects with generics — generics are monomorphized", "Forgetting that trait objects lose concrete type information", "Assuming trait objects have no runtime cost — they incur vtable indirection", "Using trait objects where generics would be more efficient"],
    },
    "generic_types": {
        "def": "types parameterized over other types, enabling the same code to work with many concrete types while preserving compile-time type safety and monomorphization",
        "principles": ["Generics are parameterized types and functions", "The compiler monomorphizes generics by generating code for each concrete type", "Generics have zero runtime cost compared to specialized code", "Type parameters can have trait bounds constraining their behavior", "Generic code is checked once, then used with any compatible type"],
        "examples": ["Vec<T> being generic over the element type T", "A function fn first<T>(v: &[T]) -> Option<&T> working for any slice", "Defining a generic Pair struct with two type parameters"],
        "mistakes": ["Confusing generics with trait objects — they monomorphize", "Over-constraining generics with unnecessary bounds", "Forgetting that monomorphization increases binary size", "Treating generics as dynamically dispatched — they are not", "Using generics where trait objects would simplify code"],
    },
    "where_clauses": {
        "def": "syntax for specifying trait bounds on generic types separately from the function signature, improving readability for complex bounds",
        "principles": ["Where clauses express bounds after the function signature", "Where clauses are equivalent to inline bounds but more readable", "Where clauses can express bounds on associated types", "Where clauses can express lifetime relationships", "Where clauses are required for some complex bounds"],
        "examples": ["fn f<T, U>() where T: Clone, U: From<T> expressing bounds separately", "Using where to bound associated types like where T::Item: Display", "Specifying multiple bounds in a single where clause"],
        "mistakes": ["Using where clauses when inline bounds would be clearer", "Forgetting that where clauses are syntactic sugar, not a different feature", "Confusing where clauses with feature gates", "Adding redundant bounds already implied by other bounds", "Misplacing the where keyword in the function signature"],
    },
    "associated_types": {
        "def": "types defined as part of a trait that the implementor chooses, providing a single output type per implementation and avoiding the need to specify the type at every use site",
        "principles": ["Associated types are declared in traits and chosen per implementation", "Each impl picks exactly one associated type value", "Associated types improve ergonomics versus generic type parameters", "Associated types can have bounds in the trait definition", "Associated types enable expressing output type relationships"],
        "examples": ["Iterator trait declaring type Item that implementations choose", "Deref trait declaring type Target that the deref target type", "Add trait declaring type Output for the result type"],
        "mistakes": ["Confusing associated types with generic type parameters", "Forgetting to specify the associated type when calling trait methods", "Trying to have multiple implementations with different associated types for the same type", "Overusing associated types when generic parameters would be clearer", "Forgetting that associated types can be constrained in where clauses"],
    },
    "default_impls": {
        "def": "method bodies provided in trait definitions that implementations can use as-is or override, enabling building rich interfaces from a small number of required methods",
        "principles": ["Default implementations can call other trait methods", "Default methods can be overridden by implementors", "Defaults enable ergonomic trait implementation", "Defaults can reduce boilerplate for common cases", "Defaults can evolve over time without breaking implementors"],
        "examples": ["Iterator::map providing a default that uses next()", "Default methods in Display enabling print formatting with optional customization", "Builder traits providing default setters that can be specialized"],
        "mistakes": ["Overriding default methods when not necessary", "Forgetting that default methods can call other trait methods", "Treating defaults as required — they are optional to override", "Adding defaults that change semantics in subtle ways", "Confusing default trait methods with default type arguments"],
    },
    "trait_bounds": {
        "def": "constraints on generic type parameters specifying the traits they must implement, ensuring generic code can only use the methods those traits guarantee",
        "principles": ["Bounds restrict what types can instantiate a generic", "Bounds enable calling trait methods on generic types", "Multiple bounds can be combined with + syntax", "Bounds can be specified inline or in where clauses", "Bounds are checked at compile time"],
        "examples": ["fn print<T: Display>(x: T) requiring Display", "fn clone_all<T: Clone>(v: &[T]) -> Vec<T> requiring Clone", "Combining bounds like T: Clone + Ord"],
        "mistakes": ["Adding bounds not actually needed by the function body", "Forgetting bounds and getting unhelpful error messages", "Combining too many bounds reducing generic flexibility", "Confusing bounds with associated type constraints", "Specifying bounds redundantly in both signature and where clause"],
    },
    "newtype_pattern": {
        "def": "wrapping a type in a single-field tuple struct to give it a distinct identity, enabling the orphan rule, adding type safety, and allowing trait implementations without modifying the wrapped type",
        "principles": ["Newtypes have zero runtime cost", "Newtypes enable implementing foreign traits on foreign types", "Newtypes prevent type confusion between semantically distinct uses", "Newtypes can carry additional validation in their constructors", "Deref can ease migration between the wrapped and unwrapped types"],
        "examples": ["struct Meters(f64); distinguishing meters from other float quantities", "struct UserId(String); preventing mixing UserId with other String-based ids", "struct Years(i64); enabling custom Display and arithmetic for years"],
        "mistakes": ["Adding unnecessary complexity with newtypes for simple cases", "Forgetting that newtype fields are private by default", "Confusing newtype with type aliases — they are different", "Overusing Deref to mimic inheritance — it is discouraged", "Skipping validation in newtype constructors"],
    },
    "phantom_types": {
        "def": "zero-sized type parameters used to encode type-level information without carrying runtime data, often via PhantomData<T> markers",
        "principles": ["PhantomData<T> has zero runtime size", "PhantomData expresses type relationships for variance and drop checking", "PhantomData can encode ownership relationships for safety", "PhantomData is erased at runtime", "PhantomData affects variance and drop behavior"],
        "examples": ["Using PhantomData<&'a T> in a struct to express a lifetime relationship", "Using PhantomData<T> in a type that doesn't store T but acts as if it does", "Marker types like PhantomData<*const T> for variance without ownership"],
        "mistakes": ["Using PhantomData where a real field would be clearer", "Forgetting that PhantomData affects drop checking", "Confusing PhantomData with the marker trait pattern", "Overcomplicating types with excessive phantom parameters", "Forgetting that PhantomData<*const T> implies variance behavior"],
    },
    "drop_trait": {
        "def": "the trait providing a drop method called automatically when a value goes out of scope, enabling deterministic resource cleanup",
        "principles": ["Drop::drop is called automatically when values go out of scope", "Drop is called in reverse order of declaration", "Implementing Drop prevents moving out of fields", "Drop cannot be called explicitly; std::mem::drop can", "Drop is deterministic and runs at known program points"],
        "examples": ["Implementing Drop for a custom file handle to close on scope exit", "RAII guards for mutexes unlocking via Drop", "Custom allocation wrappers freeing memory in Drop"],
        "mistakes": ["Trying to call drop directly instead of std::mem::drop", "Forgetting that Drop prevents partial moves from the struct", "Panicking in drop causing double-panic aborts", "Implementing Drop when not necessary — usually it isn't", "Assuming Drop is the same as destructors in C++ — semantics differ"],
    },
    "deref_trait": {
        "def": "a trait enabling the dereference operator * to work on smart pointers and wrappers by providing a method returning a reference to the inner data, with implicit deref coercion for ergonomic API use",
        "principles": ["Deref::deref returns a reference to the target type", "Deref coercion converts &Wrapper to &Inner automatically", "DerefMut enables mutable deref coercion", "Deref should only be used for smart pointers, not arbitrary wrappers", "Deref chains enable multi-level dereferencing"],
        "examples": ["String derefs to str enabling &str methods on String", "Box<T> derefs to T enabling method calls on Boxed values", "Vec<T> derefs to [T] enabling slice methods on Vecs"],
        "mistakes": ["Using Deref to mimic inheritance — it is discouraged", "Implementing Deref for non-pointer-like types", "Forgetting that DerefMut is needed for mutable coercion", "Assuming Deref coercion works for all types — it requires impl", "Confusing Deref with AsRef — they have different semantics"],
    },
    "from_into_traits": {
        "def": "traits providing infallible (From) and fallible (TryFrom) conversions between types, enabling ergonomic and explicit type transformations with clear semantics",
        "principles": ["From<T> for U implies Into<U> for T automatically", "From is infallible; TryFrom returns Result", "Implementing From is preferred over implementing Into directly", "From/Into enable the .into() method for conversions", "From<&str> for String provides string conversion"],
        "examples": ["Implementing From<i32> for MyNumber wrapping the integer", "Using .into() to convert String into MyString wrapper", "Implementing TryFrom<&str> for MyType parsing strings"],
        "mistakes": ["Implementing Into directly instead of From — From is preferred", "Confusing From with AsRef — they have different semantics", "Using From for fallible conversions — use TryFrom instead", "Forgetting that implementing From gives Into for free", "Overusing conversions when methods would be clearer"],
    },
    "iterators": {
        "def": "the Iterator trait providing a lazy sequence abstraction with a next method, enabling composable functional-style pipelines over collections and other sources",
        "principles": ["Iterators are lazy — they compute on demand", "Iterator::next returns Option<Item>", "Iterator adapters return new iterators, composing into pipelines", "Consumers like collect drive iteration and produce final results", "Iterators are zero-cost when monomorphized and inlined"],
        "examples": ["Iterating over a Vec with .iter() yielding references", "Using .map().filter().collect() to transform and filter data", "Calling .next() manually on an iterator for fine control"],
        "mistakes": ["Confusing .iter() and .into_iter() — ownership differs", "Forgetting that .collect() needs a type annotation sometimes", "Holding onto iterators past the source collection lifetime", "Assuming iterators always own their data — most borrow", "Creating infinite iterators without using .take()"],
    },
    "iterator_adapters": {
        "def": "methods on Iterator that wrap an iterator and produce a new one with modified behavior, such as map, filter, take, skip, zip, flat_map, and enumerate, enabling declarative pipelines",
        "principles": ["Adapters return new iterators lazily", "Adapters compose into pipelines", "Adapters are zero-cost abstractions when compiled", "Each adapter has specific semantics for ordering and side effects", "Adapters can be used to build complex transformations declaratively"],
        "examples": [".map(|x| x * 2) doubling each element", ".filter(|x| *x > 0) keeping only positive elements", ".flat_map(|x| x.into_iter()) flattening nested iterators"],
        "mistakes": ["Calling .collect() too early breaking laziness", "Forgetting that .map() is lazy — nothing happens until consumed", "Confusing .take(n) with .skip(n)", "Using .fold() when .sum() or .product() would be clearer", "Modifying the source during iteration causing undefined behavior"],
    },
    "collect_into": {
        "def": "the ability of Iterator::collect to gather elements into any type implementing FromIterator, including Vec, HashMap, String, and Result<Vec<T>, E>",
        "principles": ["Collect requires the target type to implement FromIterator", "Type inference often needs a hint to pick the target type", "Collecting into Result<Vec<T>, E> short-circuits on first error", "Collecting into String works for iterators of char or &str", "Collect is the standard way to materialize iterator pipelines"],
        "examples": [".collect::<Vec<_>>() gathering into a Vec", ".collect::<Result<Vec<_>, _>>() short-circuiting on errors", ".collect::<HashMap<_, _>>() building a HashMap from pairs"],
        "mistakes": ["Forgetting to specify the target type leading to inference errors", "Collecting into a type that doesn't implement FromIterator", "Using collect when a specific consumer like sum would be clearer", "Forgetting that Result collect short-circuits on first error", "Collecting into a Vec when a more efficient collection would do"],
    },
    "async_await": {
        "def": "Rust's syntax for writing asynchronous code using async fn to define futures and .await to suspend until a future completes, compiled into state machines without runtime threads",
        "principles": ["async fn returns a Future, not the result directly", ".await suspends the future until ready, without blocking threads", "Futures in Rust are lazy — they do nothing until polled", "async/await compiles to state machines, not threads", "Pinning is required to safely poll futures"],
        "examples": ["async fn fetch(url: &str) -> Result<String, Error> defining an async function", "let resp = fetch(url).await?; awaiting a future", "Spawning tasks with tokio::spawn running futures concurrently"],
        "mistakes": ["Forgetting to await a future — it does nothing", "Awaiting in a sync context blocking the thread", "Treating async fn as parallel by default — it is concurrent only if spawned", "Holding non-Send futures across await points in multi-threaded runtimes", "Forgetting that futures are lazy and need an executor"],
    },
    "futures": {
        "def": "the Future trait representing a value that may not be available yet, with a poll method that the executor calls to check progress, enabling composable asynchronous computation",
        "principles": ["Futures are lazy and must be polled to make progress", "Poll returns Pending or Ready", "Wakers notify executors when to re-poll", "Futures compose via combinators like then, map, and join", "Futures must be pinned to be polled safely"],
        "examples": ["Implementing Future for a custom async operation", "Composing futures with future::join_all running them concurrently", "Returning Box<dyn Future> for type-erased async returns"],
        "mistakes": ["Polling futures manually instead of using async/await", "Forgetting to register the waker causing the future to never resume", "Treating futures as threads — they are not", "Confusing Pending with an error condition", "Forgetting that futures must be pinned for safe polling"],
    },
    "pinning": {
        "def": "the Pin<P> wrapper ensuring values pointed to by P cannot be moved in memory, required for self-referential futures and other types that would be unsafe if moved",
        "principles": ["Pin prevents safe code from moving the pinned value", "Pinning is required for self-referential types like async state machines", "Pin<P> where P: DerefMut gives access to &mut T but not move", "Unpin types can be safely moved even when pinned", "Stack vs heap pinning has different safety requirements"],
        "examples": ["Pin<Box<dyn Future>> pinning a future on the heap", "Pin<&mut T> for polling futures", "Box::pin converting a future into a pinned Box"],
        "mistakes": ["Treating pinning as optional for async/await — it is required", "Confusing Pin with general immutability — they differ", "Forgetting that Unpin types ignore most pinning rules", "Using Pin for types that don't need it", "Assuming Pin guarantees the data won't be dropped — it doesn't"],
    },
    "tokio_runtime": {
        "def": "an asynchronous runtime for Rust providing a multi-threaded or current-thread executor, async I/O, timers, and synchronization primitives for building async applications",
        "principles": ["Tokio provides an executor that drives futures", "Multi-threaded runtime schedules tasks across worker threads", "Tokio I/O integrates with the runtime for non-blocking operations", "Tokio includes timers, channels, and synchronization primitives", "Spawning tasks requires Send futures on multi-threaded runtime"],
        "examples": ["#[tokio::main] marking an async main function", "tokio::spawn running a future concurrently as a task", "tokio::time::sleep awaiting a timer"],
        "mistakes": ["Using blocking operations inside async functions blocking the runtime", "Forgetting to make futures Send when spawning", "Spawning too many tasks without backpressure", "Confusing current-thread and multi-threaded runtimes", "Forgetting to await spawned tasks if you need their result"],
    },
    "channels_mpsc": {
        "def": "multi-producer single-consumer channels providing bounded or unbounded queues for sending values from multiple senders to a single receiver, commonly used for task communication",
        "principles": ["MPSC channels have many senders and one receiver", "Bounded channels apply backpressure on senders", "Unbounded channels grow without limit", "Senders can be cloned; receivers cannot", "Receivers implement Stream for async use"],
        "examples": ["tokio::sync::mpsc::channel creating a bounded channel", "Cloning tx and moving to multiple tasks to send concurrently", "Awaiting rx.recv() to receive messages in the consumer"],
        "mistakes": ["Using unbounded channels without backpressure causing OOM", "Forgetting that dropping all senders closes the channel", "Holding locks across .send().await causing deadlocks", "Assuming receivers are Clone — they are not", "Forgetting that bounded send awaits when full"],
    },
    "channels_spsc": {
        "def": "single-producer single-consumer channels optimized for one sender and one receiver, often used in embedded and high-throughput data pipelines",
        "principles": ["SPSC channels have one sender and one receiver", "SPSC enables lock-free implementations", "SPSC is common in real-time and embedded systems", "SPSC channels cannot be cloned", "SPSC offers lower overhead than MPSC"],
        "examples": ["crossbeam::channel::single creating an SPSC channel", "heapless::spsc for embedded SPSC queues", "Using SPSC to connect a producer thread to a consumer thread"],
        "mistakes": ["Trying to clone SPSC senders or receivers", "Using MPSC when SPSC would be more efficient", "Forgetting that SPSC has stricter single-ownership rules", "Assuming SPSC implies bounded — implementations vary", "Confusing sync and async SPSC semantics"],
    },
    "arc_mutex": {
        "def": "the combination of Arc (Atomic Reference Counted) for shared ownership across threads and Mutex for exclusive access, enabling safe shared mutable state in concurrent Rust",
        "principles": ["Arc enables multiple owners across threads via atomic refcount", "Mutex provides exclusive access via locking", "Arc<Mutex<T>> is the standard pattern for shared mutable state", "Mutex::lock returns a Result because of potential poisoning", "RwLock allows multiple readers or one writer"],
        "examples": ["Arc<Mutex<Vec<T>>> sharing a vector across threads", "Cloning Arc to give each thread its own handle", "Acquiring the lock with mutex.lock().unwrap()"],
        "mistakes": ["Forgetting to lock the mutex before accessing inner data", "Holding locks across await points causing deadlocks", "Using Mutex where RwLock would be more efficient for read-heavy workloads", "Forgetting that Mutex poisoning requires .unwrap() handling", "Confusing Arc with Rc — Rc is not thread-safe"],
    },
    "send_sync": {
        "def": "marker traits guaranteeing that types can be safely transferred (Send) or shared (Sync) across threads, automatically derived by the compiler for types whose components are Send/Sync",
        "principles": ["Send means a type can be moved across threads", "Sync means a type can be shared by reference across threads", "Most types are Send and Sync automatically", "Rc is not Send; Arc is Send if T is Send and Sync", "Cell and RefCell are not Sync"],
        "examples": ["Arc<T> being Send+Sync when T is Send+Sync", "Rc<T> not being Send forcing single-threaded use", "Mutex<T> being Sync when T is Send"],
        "mistakes": ["Confusing Send and Sync — they are different guarantees", "Trying to use Rc across threads — use Arc instead", "Forgetting that Cell and RefCell are not Sync", "Assuming all generic types are Send+Sync — they aren't unless bounded", "Forgetting that Raw pointers are neither Send nor Sync by default"],
    },
    "rayon_parallelism": {
        "def": "a data parallelism library providing parallel iterators that automatically split work across a thread pool, enabling effortless parallelization of iterator pipelines",
        "principles": ["Rayon provides a work-stealing thread pool", "par_iter() is the parallel equivalent of iter()", "Rayon preserves iterator adapter composition", "Rayon tasks must be Send and Sync", "Rayon enables data parallelism without explicit thread management"],
        "examples": ["vec.par_iter().map(|x| x * 2).collect() parallel mapping", "par_iter().for_each(|x| process(x)) parallel iteration", "par_chunks(chunk_size) processing data in parallel chunks"],
        "mistakes": ["Using par_iter when sequential iter would be faster for small data", "Forgetting that closures must be Send+Sync", "Holding locks across par_iter causing contention", "Assuming par_iter is always faster — overhead can dominate", "Forgetting that par_iter consumes the iterator differently than iter"],
    },
    "error_handling_result": {
        "def": "Rust's primary error handling mechanism using the Result<T, E> enum to represent either success (Ok) or failure (Err), with the ? operator for ergonomic propagation",
        "principles": ["Result<T, E> explicitly represents fallibility", "The ? operator propagates errors concisely", "Result forces callers to handle errors", "Custom error types enable domain-specific error handling", "Result is preferred over exceptions for predictable control flow"],
        "examples": ["Returning Result<String, io::Error> from a function reading a file", "Using ? to propagate errors: let s = fs::read_to_string(path)?;", "Matching on Result to handle errors explicitly"],
        "mistakes": ["Using .unwrap() instead of proper error handling in production", "Forgetting that ? requires the function to return Result", "Using String as the error type instead of a typed error", "Swallowing errors with .ok() when they should be propagated", "Confusing Result with Option — they have different semantics"],
    },
    "anyhow_thiserror": {
        "def": "complementary error libraries where anyhow provides dynamic, context-rich error handling for applications and thiserror provides derive macros for ergonomic custom error types in libraries",
        "principles": ["anyhow::Error is a trait object wrapping any error type", "anyhow adds context with .context() and .with_context()", "thiserror::Error derives custom error enums with Display and From", "thiserror is for libraries; anyhow is for applications", "anyhow and thiserror compose well"],
        "examples": ["#[derive(thiserror::Error)] defining a library error enum", "Returning anyhow::Result from application code", "Adding context via file.context(format!(\"failed to open {path}\"))?"],
        "mistakes": ["Using anyhow in libraries — thiserror is preferred for libraries", "Forgetting to add context losing the failure path", "Treating anyhow::Error as a single concrete type — it's a wrapper", "Using thiserror when anyhow would suffice", "Forgetting that anyhow::Error dyn-dispatches, costing performance"],
    },
    "unsafe_rust": {
        "def": "a subset of Rust where the compiler's safety guarantees are manually upheld by the programmer, used for FFI, raw pointer manipulation, and other low-level operations",
        "principles": ["unsafe enables five superpowers: deref raw pointers, call unsafe functions, access mutable statics, implement unsafe traits, access union fields", "unsafe does not turn off the borrow checker", "unsafe code must uphold Rust's safety invariants manually", "unsafe should be encapsulated in safe abstractions", "unsafe blocks document where invariants must be manually verified"],
        "examples": ["Calling libc functions through FFI requiring unsafe", "Dereferencing raw pointers for low-level memory manipulation", "Implementing unsafe trait Send for a type with raw pointers"],
        "mistakes": ["Treating unsafe as a way to bypass the borrow checker — it isn't", "Spreading unsafe widely instead of encapsulating it", "Forgetting that unsafe code must uphold invariants for safe callers", "Using unsafe when safe alternatives exist", "Not documenting why unsafe is needed and what invariants are required"],
    },
    "ffi": {
        "def": "Foreign Function Interface enabling Rust code to call and be called by C libraries, using extern blocks, repr(C), and unsafe for cross-language interop",
        "principles": ["extern \"C\" declares foreign functions", "#[repr(C)] ensures C-compatible struct layout", "FFI calls require unsafe because the compiler cannot verify C code", "C types map to specific Rust types via std::os::raw or c_char etc", "Ownership across FFI must be managed carefully to avoid leaks or double frees"],
        "examples": ["Calling libc::malloc and free through FFI", "Exposing a Rust function with #[no_mangle] pub extern \"C\" fn", "Using bindgen to auto-generate Rust bindings from C headers"],
        "mistakes": ["Using Rust types directly in FFI — use C-compatible types", "Forgetting #[no_mangle] when exporting functions", "Letting Rust panic across FFI causing undefined behavior", "Confusing ownership across the boundary", "Not handling C strings correctly — they are null-terminated"],
    },
    "procedural_macros": {
        "def": "Rust macros that run at compile time generating Rust code, including derive macros, attribute macros, and function-like macros, enabling metaprogramming and code generation",
        "principles": ["Procedural macros consume TokenStream and produce TokenStream", "Derive macros add trait implementations", "Attribute macros annotate items", "Function-like macros look like function calls but generate code", "Procedural macros must be in their own crate with proc-macro = true"],
        "examples": ["#[derive(Debug)] generating Debug impl", "#[serde(rename_all = \"kebab-case\")] transforming struct definitions", "sqlx::query! generating checked SQL queries at compile time"],
        "mistakes": ["Confusing declarative macros (macro_rules!) with procedural macros", "Forgetting that proc-macros must be in a separate crate", "Treating proc-macros as runtime code — they are compile-time", "Overusing proc-macros when simpler code would do", "Generating invalid tokens causing confusing compile errors"],
    },
}
SEEDS.update(RUST_SEEDS)

# === PYTHON SEEDS (35) ===========================
PYTHON_SEEDS = {
    "generators": {
        "def": "functions that use the yield keyword to produce a sequence of values lazily, suspending execution between yields and resuming on the next iteration, enabling memory-efficient streaming pipelines",
        "principles": ["Generators use yield to produce values one at a time", "Generators are lazy — they compute on demand", "Generators maintain local state between yields", "Generators implement the iterator protocol", "Generators can be infinite without consuming memory"],
        "examples": ["def count() yielding integers forever", "A generator reading a large file line by line without loading it all", "A pipeline of generators transforming a stream of records"],
        "mistakes": ["Treating generator functions as eagerly evaluated", "Reusing a generator after exhaustion yielding nothing", "Confusing yield with return — they have different semantics", "Forgetting that exceptions in generators can be swallowed", "Holding resources in a generator without proper cleanup"],
    },
    "decorators": {
        "def": "callables that take a function and return a replacement, applied via the @ syntax, used for wrapping behavior with cross-cutting concerns like logging, caching, and access control",
        "principles": ["Decorators are applied with the @ syntax", "Decorators can take arguments via nested functions", "functools.wraps preserves the wrapped function's metadata", "Decorators can be stacked — applied bottom-up", "Class-based decorators use __call__ and __init__"],
        "examples": ["@property turning a method into a property accessor", "@lru_cache memoizing function results", "@app.route('/path') registering a Flask view"],
        "mistakes": ["Forgetting functools.wraps losing function metadata", "Confusing decorators with no arguments vs with arguments", "Not returning a callable from the decorator", "Stacking decorators in the wrong order", "Using decorators where simple functions would suffice"],
    },
    "context_managers": {
        "def": "objects defining setup and teardown behavior for use in with statements, ensuring resources are properly acquired and released even when exceptions occur",
        "principles": ["Context managers define __enter__ and __exit__", "The with statement guarantees __exit__ is called", "contextlib.contextmanager simplifies writing context managers with generators", "Context managers can be nested with multiple with statements or ExitStack", "__exit__ receives exception info for cleanup decisions"],
        "examples": ["open(path) as a context manager for file handling", "contextlib.suppress catching specific exceptions", "Custom context manager for database transactions"],
        "mistakes": ["Forgetting that __exit__ should not re-raise unless needed", "Not handling exceptions in __exit__ properly", "Using try/finally where a context manager would be cleaner", "Confusing context managers with decorators", "Returning True from __exit__ swallowing exceptions unintentionally"],
    },
    "descriptors": {
        "def": "objects defining __get__, __set__, and/or __delete__ methods that customize attribute access on instances, powering features like properties, classmethods, and staticmethods",
        "principles": ["Descriptors define __get__ and optionally __set__", "Data descriptors override instance __dict__; non-data don't", "Properties are descriptors built into Python", "Class-level descriptors are shared across instances", "Descriptors enable reusable attribute behavior"],
        "examples": ["property implemented as a descriptor", "A typed field descriptor validating attribute types", "cached_property using a descriptor to lazy-compute and cache"],
        "mistakes": ["Defining descriptors on instances rather than classes", "Forgetting that non-data descriptors lose to instance __dict__", "Confusing descriptors with properties — properties are descriptors", "Not handling the instance is None case in __get__", "Overcomplicating code with descriptors when simpler patterns suffice"],
    },
    "metaclasses": {
        "def": "classes whose instances are classes, allowing customization of class creation via __init_subclass__, __new__, or __prepare__, used for framework-level patterns like registration and validation",
        "principles": ["Metaclasses subclass type", "Metaclasses customize class creation", "__init_subclass__ is often a simpler alternative", "type is the default metaclass for all classes", "Metaclasses propagate to subclasses"],
        "examples": ["A metaclass that auto-registers subclasses by name", "ABC metaclass enforcing abstract method implementation", "A metaclass that validates class attribute names"],
        "mistakes": ["Using metaclasses when simpler patterns would work", "Confusing metaclass conflicts in multiple inheritance", "Forgetting that metaclass affects all subclasses", "Treating metaclasses as a routine tool — they are for advanced use", "Not considering __init_subclass__ as a simpler alternative"],
    },
    "asyncio": {
        "def": "Python's standard library for writing concurrent code using async/await syntax, providing an event loop, coroutines, tasks, and I/O primitives for single-threaded concurrency",
        "principles": ["asyncio runs an event loop driving coroutines", "async def defines a coroutine function", "await suspends a coroutine until an awaitable completes", "Tasks enable concurrent execution of coroutines", "Blocking calls defeat asyncio — use async equivalents"],
        "examples": ["asyncio.run(main()) running the top-level coroutine", "asyncio.gather(*tasks) running coroutines concurrently", "asyncio.sleep replacing time.sleep in async code"],
        "mistakes": ["Calling blocking I/O in async code blocking the event loop", "Forgetting to await coroutines — they do nothing otherwise", "Mixing asyncio with threading carelessly", "Creating coroutines without scheduling them", "Confusing concurrency with parallelism — asyncio is single-threaded"],
    },
    "event_loop": {
        "def": "the central execution engine of asyncio that schedules and runs coroutines, callbacks, and I/O operations on a single thread, multiplexing many tasks efficiently",
        "principles": ["The event loop schedules and runs coroutines", "I/O multiplexing via selectors underlies the loop", "Tasks are coroutines wrapped for scheduling", "Callbacks can be scheduled with call_soon, call_later", "Custom event loops can be implemented and registered"],
        "examples": ["asyncio.get_event_loop() obtaining the running loop", "loop.run_until_complete(future) running a future to completion", "loop.call_later(1.0, callback) scheduling a callback"],
        "mistakes": ["Calling loop methods from outside the loop thread", "Forgetting that the loop is single-threaded", "Mixing different event loops across threads", "Holding the loop globally when asyncio.run is cleaner", "Forgetting to close the loop when manually managed"],
    },
    "coroutines": {
        "def": "functions defined with async def that can be suspended and resumed, enabling cooperative concurrency by yielding control back to the event loop at await points",
        "principles": ["Coroutines are defined with async def", "Coroutines must be awaited to execute", "Coroutines can await other awaitables", "Coroutines enable cooperative concurrency", "Coroutines are not parallel by default — they need tasks"],
        "examples": ["async def fetch(url) defining a coroutine", "result = await coroutine() awaiting a coroutine", "Coroutines composed with asyncio.gather running concurrently"],
        "mistakes": ["Forgetting to await a coroutine — it returns a coroutine object, not a result", "Treating coroutines as parallel — they are concurrent", "Using return in async generators instead of yield", "Awaiting coroutines from synchronous code", "Confusing coroutines with threads or processes"],
    },
    "tasks_futures": {
        "def": "asyncio.Task wraps a coroutine for concurrent execution on the event loop, while Future is a low-level awaitable representing a result that may not yet be available",
        "principles": ["Tasks schedule coroutines for concurrent execution", "Futures are low-level awaitables for results", "asyncio.create_task schedules a coroutine as a task", "Tasks can be awaited, cancelled, or gathered", "Futures are typically used by library code, not directly"],
        "examples": ["asyncio.create_task(coroutine()) scheduling a task", "task.result() awaiting and retrieving the result", "asyncio.gather(*tasks) running multiple tasks concurrently"],
        "mistakes": ["Forgetting to keep references to tasks — they can be garbage collected", "Confusing Tasks with coroutines — tasks wrap coroutines for scheduling", "Not handling task exceptions causing silent failures", "Cancelling tasks without proper cleanup", "Creating tasks but never awaiting them"],
    },
    "async_comprehensions": {
        "def": "syntax for creating lists, sets, and dicts from async iterables using async for inside comprehension syntax, with await allowed in expressions for async generators",
        "principles": ["Async comprehensions use async for", "Async comprehensions produce a synchronous collection, not a coroutine", "Await is allowed in async generator expressions", "Async comprehensions require async def context", "Async comprehensions simplify collecting async results"],
        "examples": ["[x async for x in async_iter] collecting into a list", "{x async for x in async_iter} collecting into a set", "{k: await f(k) for k in keys} awaiting inside a comprehension"],
        "mistakes": ["Using async for outside async def", "Forgetting await when needed in async generator expressions", "Confusing async comprehensions with sync ones — they differ", "Treating async comprehension results as awaitables", "Overusing async comprehensions when async for loops would be clearer"],
    },
    "typing_module": {
        "def": "the standard library module providing type hints and generic types like List, Dict, Optional, Callable, Union, and TypeVar, enabling static type checking",
        "principles": ["Type hints are annotations, not runtime checks", "Optional[T] is equivalent to Union[T, None]", "Callable types can be specified with Callable[[Arg], Ret]", "TypeVar enables generic type hints", "Type hints are optional but recommended for clarity"],
        "examples": ["def f(x: int) -> str: annotating a function", "List[int] annotating a list of integers", "Optional[str] = None annotating an optional string"],
        "mistakes": ["Treating type hints as runtime checks — they aren't", "Using lowercase list/dict instead of typing.List/Dict in Python <3.9", "Over-annotating where types are obvious", "Forgetting that None is the default for Optional", "Confusing type hints with runtime validation"],
    },
    "generics_pep695": {
        "def": "PEP 695 syntax for defining generic functions, classes, and type aliases using type parameters in square brackets, replacing TypeVar declarations with cleaner inline syntax",
        "principles": ["Type parameters declared with [T] after the name", "Type alias declarations use type keyword", "Generic classes use [T] in class definitions", "PEP 695 syntax requires Python 3.12+", "Type parameters are scoped to the definition"],
        "examples": ["def first[T](xs: list[T]) -> T | None: defining a generic function", "class Box[T]: defining a generic class", "type Vector = list[float] defining a type alias"],
        "mistakes": ["Using PEP 695 syntax in older Python versions", "Confusing type parameters with runtime parameters", "Forgetting that the syntax is cleaner than TypeVar but equivalent", "Mixing PEP 695 with older typing syntax unnecessarily", "Assuming type parameters are runtime-accessible — they aren't by default"],
    },
    "protocols": {
        "def": "structural subtyping via the Protocol base class, allowing types to be considered compatible based on the methods and attributes they have, without explicit inheritance",
        "principles": ["Protocols enable structural subtyping", "Protocol classes are used for type hints only, not inheritance", "Protocol methods are stubs without bodies", "@runtime_checkable enables isinstance checks", "Protocols can have default implementations"],
        "examples": ["class Iterable(Protocol) defining an iterable protocol", "Using a Protocol as a parameter type accepting any compatible type", "@runtime_checkable class Hashable(Protocol) enabling isinstance checks"],
        "mistakes": ["Treating Protocols as base classes for inheritance", "Forgetting that Protocol methods are stubs", "Using runtime checking when static checking would suffice", "Confusing structural with nominal subtyping", "Forgetting @runtime_checkable for isinstance support"],
    },
    "dataclasses": {
        "def": "a decorator and module providing automatic generation of __init__, __repr__, __eq__, and other dunder methods for classes primarily used to store data, with field descriptors for customization",
        "principles": ["@dataclass generates common dunder methods", "Fields are class annotations with optional defaults", "field() customizes field behavior", "frozen=True makes instances immutable", "Dataclasses support inheritance and slots"],
        "examples": ["@dataclass class Point: x: float; y: float defining a point", "@dataclass(frozen=True) class Config: ... defining an immutable config", "field(default_factory=list) for mutable defaults"],
        "mistakes": ["Using mutable default values directly — use default_factory", "Forgetting that fields without defaults cannot follow fields with defaults", "Treating dataclasses as enums or named tuples — they are different", "Confusing frozen with slots — they serve different purposes", "Overriding generated methods inconsistently"],
    },
    "attrs": {
        "def": "a third-party library predating and inspiring dataclasses, providing class decorators for automatically generating init, repr, eq, and other methods with rich validation and conversion features",
        "principles": ["@attr.s decorates classes to auto-generate methods", "attr.ib() defines attributes with validators and converters", "attrs supports slots, frozen, and other class-level options", "attrs pre-dates dataclasses and is more feature-rich", "attrs works on Python 2.7+ and 3.x"],
        "examples": ["@attr.s class Point: x = attr.ib(); y = attr.ib()", "attr.ib(validator=attr.validators.instance_of(int)) adding validation", "@attr.s(frozen=True, slots=True) for immutable, memory-efficient classes"],
        "mistakes": ["Confusing attrs with dataclasses — they differ in features", "Using attrs when stdlib dataclasses would suffice", "Forgetting that attrs auto-generates __init__ unless init=False", "Mixing attr.ib() with class annotations incorrectly", "Treating attrs as a replacement for all classes — it's for data containers"],
    },
    "pydantic": {
        "def": "a data validation library using Python type hints to define models with automatic validation, parsing, and serialization, popular for configuration, API schemas, and ORM-like patterns",
        "principles": ["Pydantic models use type annotations for validation", "Models validate input on instantiation", "Models serialize to dict and JSON", "Field constraints via Field() or Annotated", "Pydantic v2 is significantly faster than v1"],
        "examples": ["class User(BaseModel): id: int; name: str defining a validated model", "Field(gt=0) constraining an integer field", "model_dump() and model_dump_json() serializing models"],
        "mistakes": ["Confusing Pydantic models with dataclasses — they validate, not just store", "Using mutable defaults without Field(default_factory=...)", "Forgetting that validation runs on instantiation", "Mixing v1 and v2 APIs", "Treating Pydantic as a database ORM — it's a validation library"],
    },
    "slots_memory": {
        "def": "the __slots__ class attribute that fixes the set of allowed instance attributes, preventing __dict__ creation and significantly reducing per-instance memory usage",
        "principles": ["__slots__ prevents per-instance __dict__ creation", "__slots__ reduces memory per instance significantly", "Subclasses must also define __slots__ to keep the benefit", "__slots__ disables the default __weakref__ unless added explicitly", "Access to undeclared attributes raises AttributeError"],
        "examples": ["class Point: __slots__ = ('x', 'y') defining a memory-efficient point", "Adding __weakref__ to slots for weak reference support", "Using slots with @dataclass(slots=True) for ergonomic slot declaration"],
        "mistakes": ["Forgetting that subclasses must redefine __slots__ to preserve benefits", "Trying to add attributes not in __slots__", "Confusing __slots__ with immutability — slots can still be mutable", "Forgetting to add __weakref__ when weak references are needed", "Overusing slots for classes with few instances"],
    },
    "weakrefs": {
        "def": "references to objects that do not prevent garbage collection, allowing caches, observers, and other patterns to refer to objects without causing memory leaks",
        "principles": ["Weak references don't increment the reference count", "Weak references return None if the referent was collected", "weakref.ref creates a basic weak reference", "weakref.WeakValueDictionary and WeakKeyDictionary for caches", "Some types like list and dict don't support weakrefs directly"],
        "examples": ["weakref.ref(obj) creating a basic weak reference", "WeakValueDictionary for caching without preventing collection", "weakref.finalize registering cleanup callbacks"],
        "mistakes": ["Forgetting to check if the referent is still alive", "Using weakrefs on types that don't support them", "Confusing weakref.ref with weakref.proxy", "Holding strong references elsewhere defeating the purpose", "Assuming weakrefs prevent finalization — they don't"],
    },
    "gc_module": {
        "def": "the standard library module controlling Python's cyclic garbage collector, used to detect and collect reference cycles that reference counting alone cannot handle",
        "principles": ["Python uses reference counting plus a cyclic GC", "The cyclic GC handles reference cycles", "gc.collect() forces a collection cycle", "gc.disable() turns off the cyclic GC", "Generational GC has three generations"],
        "examples": ["gc.collect() forcing garbage collection", "gc.get_objects() inspecting tracked objects", "Disabling GC and using weakrefs for performance-critical code"],
        "mistakes": ["Disabling GC when cycles are still possible causing leaks", "Confusing reference counting with cyclic GC", "Overusing gc.collect causing performance issues", "Treating GC as a substitute for proper resource management", "Forgetting that finalizers can resurrect objects"],
    },
    "interning": {
        "def": "the practice of storing a single copy of frequently-used immutable objects, applied automatically by Python to small integers and short strings to save memory and speed comparisons",
        "principles": ["Small integers (-5 to 256) are interned by default", "Identifier-like strings are interned automatically", "sys.intern() explicitly interns strings", "Interned objects can be compared by identity", "Interning reduces memory for repeated values"],
        "examples": ["a is b returning True for small integers", "sys.intern('hello') explicitly interning a string", "String literals in code often being interned"],
        "mistakes": ["Relying on automatic interning for all strings — only some are interned", "Using is for string comparison instead of ==", "Interning short-lived strings causing overhead", "Confusing interning with caching", "Assuming interning applies to mutable objects — it doesn't"],
    },
    "import_system": {
        "def": "the mechanism by which Python locates, loads, and initializes modules, using finders to locate modules and loaders to execute them, customizable via sys.meta_path and import hooks",
        "principles": ["Importers use finders to locate modules", "Loaders execute module code and populate attributes", "sys.meta_path contains the meta path finders", "Modules are cached in sys.modules", "Custom finders can extend the import system"],
        "examples": ["importlib.import_module(name) dynamically importing a module", "Custom finder loading modules from a database", "sys.modules caching preventing re-imports"],
        "mistakes": ["Modifying sys.modules directly causing inconsistency", "Forgetting that imports are cached in sys.modules", "Confusing finders with loaders — they have different roles", "Using __import__ directly instead of importlib", "Forgetting that circular imports can cause issues"],
    },
    "namespace_packages": {
        "def": "packages split across multiple directories, allowing libraries to be distributed as separate components under a shared top-level package name, used by large projects like zope and Google's packages",
        "principles": ["Namespace packages don't require __init__.py", "PEP 420 implicit namespace packages are the modern approach", "Multiple directories can contribute to the same namespace", "Namespace packages enable split distributions", "Legacy namespace packages used pkg_resources or pkgutil"],
        "examples": ["zope.interface and zope.component as separate distributions under zope", "A company package split across multiple internal repositories", "Using find_namespace_packages in setup.py for packaging"],
        "mistakes": ["Forgetting that namespace packages differ from regular packages", "Mixing legacy and modern namespace approaches", "Adding __init__.py to namespace package directories", "Confusing namespace packages with regular subpackages", "Forgetting that namespace packages affect import resolution order"],
    },
    "relative_imports": {
        "def": "imports using leading dots to refer to modules in the same package, with one dot for the current package, two for the parent, and so on, simplifying package refactoring",
        "principles": ["from . import module imports from the current package", "from .. import module imports from the parent package", "Relative imports require being inside a package", "Relative imports don't work for the main module", "Absolute imports are generally preferred for clarity"],
        "examples": ["from . import helpers importing helpers from the current package", "from ..models import User importing from the parent package", "from .submodule import func importing from a submodule"],
        "mistakes": ["Using relative imports in __main__ modules causing ImportError", "Confusing single dot with double dot semantics", "Using too many dots referring beyond the top-level package", "Mixing relative and absolute imports inconsistently", "Forgetting that relative imports require running as a package"],
    },
    "dunder_methods": {
        "def": "special methods with double-underscore prefixes and suffixes like __init__, __len__, __getitem__, that Python calls implicitly to implement language features like iteration, indexing, and operator overloading",
        "principles": ["Dunder methods implement language protocols", "Python calls dunders implicitly; direct calls are discouraged", "Dunders enable operator overloading, iteration, and more", "Many dunders have default implementations", "Implementing dunders makes custom types act like built-ins"],
        "examples": ["__len__ enabling len(obj) on custom containers", "__getitem__ enabling obj[key] indexing", "__iter__ and __next__ enabling for loops over custom iterables"],
        "mistakes": ["Calling dunder methods directly instead of using the corresponding builtin", "Forgetting to return the right type from dunders", "Implementing __eq__ without __hash__ breaking dict usage", "Overloading operators in surprising ways violating expectations", "Treating dunders as private — they are part of the protocol"],
    },
    "iteration_protocols": {
        "def": "the iterator protocol consisting of __iter__ returning an iterator and __next__ returning the next value or raising StopIteration, enabling for loops and other iteration to work uniformly across types",
        "principles": ["Iterables define __iter__ returning an iterator", "Iterators define __next__ and __iter__", "StopIteration signals the end of iteration", "Generators automatically implement the iterator protocol", "Iterators are consumed once and then exhausted"],
        "examples": ["A custom range-like class implementing __iter__ and __next__", "Generators implicitly implementing the iterator protocol", "iter() builtin calling __iter__ on iterables"],
        "mistakes": ["Confusing iterables with iterators — iterables produce iterators", "Forgetting that __next__ must raise StopIteration", "Reusing exhausted iterators expecting more values", "Not implementing __iter__ on iterators causing issues with for loops", "Treating iterators as restartable"],
    },
    "slicing_advanced": {
        "def": "extended slice syntax including step values, negative indices, slice objects, and Ellipsis usage, enabling concise selection of subsequences from sequences and multidimensional arrays",
        "principles": ["slice[start:stop:step] defines a slice", "Negative indices count from the end", "Step can be negative for reverse iteration", "Slice objects can be created and reused", "Ellipsis (...) and None are used in multidimensional slicing"],
        "examples": ["a[::-1] reversing a sequence", "a[1::2] taking every other element starting from index 1", "slice(1, 5, 2) creating a reusable slice object"],
        "mistakes": ["Confusing slice stop with inclusive end — it's exclusive", "Forgetting that negative step reverses start/stop meaning", "Using slices on non-sequence types causing TypeError", "Confusing slice with index access semantics", "Forgetting that slices return new objects, not views"],
    },
    "walrus_operator": {
        "def": "the := operator introduced in PEP 572 that assigns a value as part of an expression, enabling assignment in conditions, comprehensions, and other expression contexts",
        "principles": ["The walrus operator assigns within an expression", "It reduces repetition in conditions and comprehensions", "The assignment target must be a name, not an attribute or index", "The expression evaluates to the assigned value", "Walrus operator can clarify intent in some patterns"],
        "examples": ["if (n := len(a)) > 10: using the value in the body", "[y for x in data if (y := f(x)) > 0] capturing the function call", "while (chunk := file.read(8192)): processing chunks until empty"],
        "mistakes": ["Overusing the walrus operator when simple assignment is clearer", "Using it in ways that reduce readability", "Confusing = and := semantics", "Using it where the value isn't reused", "Forgetting it's only available in Python 3.8+"],
    },
    "pattern_matching": {
        "def": "structural pattern matching introduced in PEP 634 (Python 3.10+) using match and case statements to destructure objects, sequences, mappings, and literals in a declarative style",
        "principles": ["match/case is structural pattern matching", "Patterns can include literals, captures, and wildcards", "Patterns support destructuring of sequences, mappings, and classes", "Guards with if enable additional conditions", "Matching is exhaustive only with explicit default"],
        "examples": ["match point: case (0, 0): ... matching a tuple literal", "case Point(x=0, y=y): destructuring a class instance", "case [first, *rest]: destructuring a list with capture"],
        "mistakes": ["Forgetting that patterns bind names, not test equality", "Confusing match with switch in other languages — match is more powerful", "Forgetting that _ is a wildcard, not a variable", "Not handling all cases leading to silent failures", "Using match when if-elif would be clearer for simple cases"],
    },
    "exception_chaining": {
        "def": "the mechanism by which exceptions can be linked using raise ... from ... or implicit chaining via __context__ during exception handling, preserving the original traceback",
        "principles": ["raise NewException from original_exception explicitly chains", "Implicit chaining occurs when raising in an except block", "__cause__ holds the explicit chained exception", "__context__ holds the implicit chained exception", "Chaining preserves the full error context"],
        "examples": ["raise DatabaseError(...) from original_io_error", "Implicit chaining when re-raising in an except block", "raise ... from None suppressing the chained context"],
        "mistakes": ["Forgetting to chain exceptions losing the original cause", "Using raise ... from None to suppress when chaining is helpful", "Confusing __cause__ and __context__", "Swallowing exceptions with bare except without re-raising", "Not handling chained exceptions in logging"],
    },
    "contextvars": {
        "def": "the contextvars module providing ContextVar objects that hold values scoped to a logical execution context, enabling task-local state in async code without thread-local pitfalls",
        "principles": ["ContextVar holds values per execution context", "ContextVar replaces thread-local storage in async code", "Each asyncio task gets its own copy of context variables", "contextvars.Token enables resetting to previous values", "ContextVar works across sync and async code"],
        "examples": ["request_id: ContextVar[str] = ContextVar('request_id')", "request_id.set('abc-123') setting a value", "var.get() retrieving the current value"],
        "mistakes": ["Using threading.local in async code — use ContextVar", "Forgetting that ContextVar values don't propagate across tasks by default", "Confusing ContextVar with global state — they differ", "Not resetting values via tokens leading to leaks", "Assuming ContextVar is per-thread — it's per-context"],
    },
    "concurrent_futures": {
        "def": "a high-level standard library module providing ThreadPoolExecutor and ProcessPoolExecutor for parallel execution of functions, returning Future objects to retrieve results",
        "principles": ["ThreadPoolExecutor uses threads for I/O-bound work", "ProcessPoolExecutor uses processes for CPU-bound work", "submit() returns a Future for retrieving the result", "map() parallels the builtin map across the pool", "Executor should be used as a context manager"],
        "examples": ["with ThreadPoolExecutor() as ex: results = list(ex.map(f, items))", "future = ex.submit(long_running_fn, arg); result = future.result()", "ProcessPoolExecutor for CPU-bound parallel computation"],
        "mistakes": ["Using ThreadPoolExecutor for CPU-bound work hitting the GIL", "Forgetting to call shutdown or use with statement", "Not handling exceptions from futures", "Submitting too many tasks without backpressure", "Confusing Future with asyncio.Future — they are different"],
    },
    "multiprocessing": {
        "def": "the standard library module for creating and managing processes, enabling true parallelism by bypassing the GIL, with Queue, Pipe, Pool, and shared memory primitives",
        "principles": ["multiprocessing creates separate processes", "Each process has its own Python interpreter and memory", "Pool manages a pool of worker processes", "Queue and Pipe enable inter-process communication", "SharedMemory enables efficient shared state"],
        "examples": ["multiprocessing.Pool(4) creating a 4-worker pool", "pool.map(f, items) parallel mapping across processes", "multiprocessing.Queue for inter-process communication"],
        "mistakes": ["Forgetting that processes don't share memory by default", "Passing non-picklable arguments to processes", "Not joining processes causing zombies", "Confusing multiprocessing with threading", "Forgetting that Windows uses spawn by default requiring import guards"],
    },
    "threading_gil": {
        "def": "the Global Interpreter Lock (GIL) in CPython that allows only one thread to execute Python bytecode at a time, limiting threading for CPU-bound work while still helping with I/O-bound work",
        "principles": ["The GIL allows only one thread to run Python bytecode at a time", "The GIL is released during I/O and certain C extensions", "Threading helps for I/O-bound work but not CPU-bound", "Multiprocessing bypasses the GIL", "PEP 703 discusses making the GIL optional in future Python"],
        "examples": ["I/O-bound threads running concurrently due to GIL release", "CPU-bound threads running serially due to the GIL", "C extensions releasing the GIL enabling true parallelism"],
        "mistakes": ["Using threading for CPU-bound work expecting parallelism", "Forgetting that the GIL is CPython-specific", "Assuming threads always speed up code", "Confusing GIL release with thread safety", "Treating the GIL as universal — Jython and IronPython don't have it"],
    },
    "ctypes_cffi": {
        "def": "standard and third-party libraries for calling C functions from Python: ctypes is built-in and simpler, while cffi is more powerful with C-like syntax and better performance",
        "principles": ["ctypes is in the standard library", "cffi is third-party but more powerful", "Both call C functions via shared libraries", "cffi supports inline C declarations", "Both have ABI and API modes (cffi) or different binding styles"],
        "examples": ["ctypes.CDLL('libc.so.6') loading a shared library", "cffi.dlopen('libc') loading with cffi", "cffi inline mode with cdef and set_source"],
        "mistakes": ["Using ctypes when cffi would be cleaner", "Forgetting to handle C type conversions", "Crashing Python via invalid C calls", "Confusing ABI and API modes in cffi", "Not handling memory management across the boundary"],
    },
    "c_extensions": {
        "def": "Python modules written in C using the Python C API or extension skeletons, enabling performance-critical code and access to C libraries, used by NumPy, asyncio, and many libraries",
        "principles": ["C extensions are compiled shared modules", "They use the Python C API", "C extensions can release the GIL for parallelism", "PyO3 enables writing extensions in Rust", "Cython provides a Python-like syntax for extensions"],
        "examples": ["NumPy's multiarray C extension for performance", "A custom C extension wrapping a numerical library", "Cython compiling Python-like code to C"],
        "mistakes": ["Writing C extensions when pure Python would suffice", "Forgetting to handle reference counting correctly", "Not releasing the GIL when possible", "Memory management bugs causing leaks or crashes", "Confusing C extensions with ctypes/cffi — they are different"],
    },
}
SEEDS.update(PYTHON_SEEDS)

# === MATH SEEDS (40) ===========================
MATH_SEEDS = {
    "linear_equations": {
        "def": "equations of the form ax + b = 0 (one variable) or more generally a system of linear equations in multiple variables, solvable by isolation, substitution, elimination, or matrix methods",
        "principles": ["Equivalent operations preserve solution sets: add same value, multiply by nonzero", "Isolate the variable by reversing operations", "Substitution replaces one variable with its equivalent expression", "Elimination adds equations to cancel a variable", "Systems have unique, no, or infinitely many solutions"],
        "examples": ["Solving 2x + 3 = 11 to get x = 4", "Solving a 2x2 system like x + y = 5, x - y = 1 yielding x = 3, y = 2", "Gaussian elimination on a 3x3 system in matrix form"],
        "mistakes": ["Forgetting to apply operations to both sides of the equation", "Dividing by zero accidentally when eliminating", "Dropping negative signs when distributing", "Confusing identity (infinite solutions) with contradiction (no solutions)", "Mixing up rows when doing elimination"],
    },
    "quadratic_equations": {
        "def": "equations of the form ax^2 + bx + c = 0 with a != 0, solvable by factoring, completing the square, or the quadratic formula x = (-b +/- sqrt(b^2 - 4ac)) / (2a)",
        "principles": ["The discriminant D = b^2 - 4ac determines root nature", "D > 0: two distinct real roots; D = 0: one repeated real root; D < 0: two complex conjugate roots", "Sum of roots = -b/a; product of roots = c/a (Vieta's formulas)", "Completing the square derives the quadratic formula", "Quadratics can be factored as a(x - r1)(x - r2)"],
        "examples": ["Solving x^2 - 5x + 6 = 0 by factoring to (x-2)(x-3) = 0 giving x = 2 or 3", "Using the quadratic formula on 2x^2 + 3x - 1 = 0", "Completing the square on x^2 + 6x + 5 = 0"],
        "mistakes": ["Forgetting the +/- in the quadratic formula giving only one root", "Confusing b^2 with b when computing the discriminant", "Mishandling negative coefficients in factoring", "Assuming two real roots when D < 0", "Dividing by 2a incorrectly when a is not 1"],
    },
    "polynomial_theory": {
        "def": "the study of polynomials p(x) = a_n x^n + ... + a_1 x + a_0, including roots, factorization, the Fundamental Theorem of Algebra, and polynomial identities",
        "principles": ["The Fundamental Theorem of Algebra: every non-constant polynomial has a complex root", "A degree-n polynomial has at most n roots counting multiplicity", "Polynomials factor completely over the complex numbers", "Rational Root Theorem narrows candidate rational roots", "Polynomial long division and synthetic division"],
        "examples": ["Factoring x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)", "Using the Rational Root Theorem on 2x^3 - 3x + 1", "Polynomial division of x^3 - 1 by x - 1"],
        "mistakes": ["Forgetting to count multiplicity when counting roots", "Assuming all real roots when some may be complex", "Misapplying the Rational Root Theorem", "Errors in synthetic division arithmetic", "Confusing factors with roots"],
    },
    "inequalities": {
        "def": "mathematical statements using <, <=, >, >= that compare expressions, solved by similar operations to equations but with sign-flipping rules for multiplication by negatives",
        "principles": ["Adding/subtracting preserves inequality direction", "Multiplying/dividing by positive preserves direction", "Multiplying/dividing by negative flips direction", "Solution sets are often intervals, not single points", "Compound inequalities combine multiple constraints"],
        "examples": ["Solving 2x - 5 < 7 to get x < 6", "Solving -3x > 9 flipping to x < -3", "Solving compound 1 <= 2x - 3 < 5 yielding 2 <= x < 4"],
        "mistakes": ["Forgetting to flip the inequality when multiplying by a negative", "Treating inequalities like equations losing interval solutions", "Confusing strict and non-strict inequalities", "Not graphing the solution set when required", "Multiplying both sides by an unknown sign variable"],
    },
    "exponents_logarithms": {
        "def": "exponents a^b and their inverse function log_a(b), obeying laws like a^m * a^n = a^(m+n), (a^m)^n = a^(mn), and log(xy) = log(x) + log(y)",
        "principles": ["a^m * a^n = a^(m+n); a^m / a^n = a^(m-n)", "(a^m)^n = a^(mn); (ab)^n = a^n b^n", "a^0 = 1; a^(-n) = 1/a^n; a^(1/n) is the nth root", "log_a(xy) = log_a(x) + log_a(y); log_a(x/y) = log_a(x) - log_a(y)", "log_a(x^n) = n log_a(x); change of base: log_b(x) = log_a(x) / log_a(b)"],
        "examples": ["Simplifying 2^3 * 2^4 = 2^7 = 128", "Solving 3^x = 81 yielding x = log_3(81) = 4", "Using log properties: log(100) + log(10) = log(1000) = 3"],
        "mistakes": ["Confusing a^m * a^n with (a^m)^n", "Treating (a+b)^n as a^n + b^n (wrong, except n=1)", "Forgetting that log(x) is undefined for x <= 0", "Confusing log base 10 with natural log", "Misapplying power rule to sums instead of products"],
    },
    "sequences_series": {
        "def": "ordered lists of numbers (sequences) and their sums (series), including arithmetic sequences with common difference, geometric sequences with common ratio, and convergent infinite series",
        "principles": ["Arithmetic: a_n = a_1 + (n-1)d; sum = n(a_1 + a_n)/2", "Geometric: a_n = a_1 * r^(n-1); sum = a_1(1 - r^n)/(1 - r)", "Infinite geometric series converges to a_1/(1 - r) if |r| < 1", "Convergence tests: ratio, root, integral, comparison", "Power series define functions and converge in a radius"],
        "examples": ["Sum of 1 + 2 + ... + 100 = 5050", "Geometric series 1 + 1/2 + 1/4 + ... = 2", "Arithmetic sequence 5, 8, 11, 14, ... with d = 3"],
        "mistakes": ["Confusing arithmetic and geometric sequence formulas", "Forgetting the |r| < 1 condition for infinite geometric convergence", "Misapplying convergence tests", "Indexing errors: starting at n=0 vs n=1", "Treating divergent series as if they converged"],
    },
    "complex_numbers": {
        "def": "numbers of the form a + bi where i^2 = -1, with arithmetic, polar form re^(i*theta), Euler's formula e^(i*pi) = -1, and the complex plane as geometric representation",
        "principles": ["i^2 = -1; complex numbers form a field", "Polar form: z = r(cos(theta) + i sin(theta)) = r e^(i*theta)", "Modulus |z| = sqrt(a^2 + b^2); argument theta = arctan(b/a)", "Euler's formula: e^(i*theta) = cos(theta) + i sin(theta)", "De Moivre's theorem: (cos theta + i sin theta)^n = cos(n*theta) + i sin(n*theta)"],
        "examples": ["Multiplying (1 + i)(2 - i) = 3 + i", "Converting 1 + sqrt(3)i to polar form: 2 e^(i*pi/3)", "Computing (1 + i)^4 using De Moivre's theorem"],
        "mistakes": ["Treating i as a variable rather than sqrt(-1)", "Forgetting to distribute when multiplying complex numbers", "Confusing modulus with argument", "Misapplying exponent rules to complex powers", "Errors in branch selection for complex logarithms"],
    },
    "set_theory": {
        "def": "the mathematical study of sets as collections of distinct objects, with operations union, intersection, complement, and concepts like subsets, power sets, and cardinality",
        "principles": ["Union A or B = elements in either; intersection A and B = elements in both", "Complement A^c = elements not in A (relative to universal set)", "Power set P(A) = all subsets of A; |P(A)| = 2^|A|", "De Morgan's laws: (A or B)^c = A^c and B^c; (A and B)^c = A^c or B^c", "Cartesian product A x B = all ordered pairs (a, b)"],
        "examples": ["{1,2,3} or {3,4,5} = {1,2,3,4,5}", "{1,2,3} and {2,3,4} = {2,3}", "Power set of {1,2} = {{}, {1}, {2}, {1,2}}"],
        "mistakes": ["Confusing subset with proper subset", "Forgetting the empty set in power sets", "Misapplying De Morgan's laws", "Treating sets as ordered (they're unordered)", "Confusing union with disjoint union"],
    },
    "relations_functions": {
        "def": "relations as subsets of Cartesian products and functions as relations where each input maps to exactly one output, with properties injective, surjective, bijective, and compositions",
        "principles": ["A function maps each input to exactly one output", "Injective (one-to-one): distinct inputs map to distinct outputs", "Surjective (onto): every element in codomain is hit", "Bijective = injective + surjective; has an inverse", "Composition (f o g)(x) = f(g(x))"],
        "examples": ["f(x) = x^2 is not injective on R (since f(-1) = f(1))", "f(x) = 2x + 1 is bijective from R to R", "Composition: if f(x) = x+1 and g(x) = x^2, then (f o g)(x) = x^2 + 1"],
        "mistakes": ["Confusing injective with surjective", "Forgetting that inverse exists only for bijective functions", "Composing in wrong order: f o g means f(g(x))", "Treating non-functions as functions", "Mixing up domain and codomain"],
    },
    "limits": {
        "def": "the value a function approaches as its input approaches some value, formalized by the epsilon-delta definition, foundational to calculus and continuity",
        "principles": ["lim_{x->a} f(x) = L means f(x) gets arbitrarily close to L as x approaches a", "Limits may exist or not, may be infinite, may differ from f(a)", "L'Hopital's rule resolves 0/0 and inf/inf forms", "Squeeze theorem: if g <= f <= h and lim g = lim h, then lim f equals that", "One-sided limits: left and right limits must agree for limit to exist"],
        "examples": ["lim_{x->0} sin(x)/x = 1 (squeeze theorem)", "lim_{x->0} (1+x)^(1/x) = e", "lim_{x->2} (x^2 - 4)/(x - 2) = 4 by simplification"],
        "mistakes": ["Evaluating f(a) instead of the limit", "Forgetting to check one-sided limits at discontinuities", "Misapplying L'Hopital to non-indeterminate forms", "Treating infinity as a number rather than a limit behavior", "Confusing limit existence with limit value"],
    },
    "continuity": {
        "def": "a function is continuous at a point if its limit exists, equals the function value, and the function is defined there; continuous functions have no jumps, holes, or vertical asymptotes",
        "principles": ["Continuous at a: lim_{x->a} f(x) = f(a)", "Three conditions: f(a) defined, limit exists, limit equals f(a)", "Polynomials, exponentials, trig functions are continuous on their domains", "Composition of continuous functions is continuous", "Intermediate Value Theorem applies on closed intervals"],
        "examples": ["f(x) = x^2 is continuous everywhere", "f(x) = 1/x is continuous on its domain (excluding 0)", "Piecewise function with a jump discontinuity at x = 2"],
        "mistakes": ["Confusing continuity with differentiability (differentiability implies continuity)", "Forgetting to check the three conditions at piecewise boundaries", "Treating rational functions as discontinuous at vertical asymptotes vs removable holes", "Assuming limit existence implies continuity", "Forgetting that |x| is continuous but not differentiable at 0"],
    },
    "differentiation": {
        "def": "the process of finding the derivative f'(x) = lim_{h->0} (f(x+h) - f(x))/h, representing the instantaneous rate of change and the slope of the tangent line",
        "principles": ["Power rule: d/dx x^n = n x^(n-1)", "Product rule: (fg)' = f'g + fg'", "Quotient rule: (f/g)' = (f'g - fg')/g^2", "Chain rule: (f(g(x)))' = f'(g(x)) g'(x)", "Derivative of e^x is e^x; derivative of ln(x) is 1/x"],
        "examples": ["d/dx (x^3 + 2x^2) = 3x^2 + 4x", "d/dx (sin(x)) = cos(x)", "d/dx (x^2 * cos(x)) = 2x cos(x) - x^2 sin(x) by product rule"],
        "mistakes": ["Forgetting the chain rule for composite functions", "Confusing product rule with quotient rule", "Treating (f+g)' as f' g' instead of f' + g'", "Errors with negative exponents in power rule", "Forgetting to multiply by inner derivative in chain rule"],
    },
    "chain_rule": {
        "def": "the rule for differentiating composite functions stating (f(g(x)))' = f'(g(x)) * g'(x), enabling differentiation of nested functions like sin(x^2), e^(3x), ln(cos(x))",
        "principles": ["Identify outer and inner functions", "Differentiate outer with respect to inner, then multiply by inner's derivative", "Apply recursively for multiple layers", "Leibniz notation: dy/dx = (dy/du)(du/dx)", "Works for any composition depth"],
        "examples": ["d/dx sin(x^2) = cos(x^2) * 2x", "d/dx e^(3x) = e^(3x) * 3 = 3e^(3x)", "d/dx ln(cos(x)) = (1/cos(x)) * (-sin(x)) = -tan(x)"],
        "mistakes": ["Forgetting to multiply by the inner derivative", "Differentiating outer with respect to x instead of inner", "Stopping at one layer when multiple are nested", "Confusing chain rule with product rule", "Misidentifying inner and outer functions"],
    },
    "optimization": {
        "def": "finding the maximum or minimum values of a function, often by setting the derivative to zero, classifying critical points, and checking boundary behavior",
        "principles": ["Critical points: where f'(x) = 0 or f'(x) doesn't exist", "Second derivative test: f''>0 means local min; f''<0 means local max; f''=0 inconclusive", "First derivative test: sign change of f' indicates extremum", "Closed interval method: check critical points and endpoints", "Concavity: f''>0 concave up; f''<0 concave down"],
        "examples": ["Finding dimensions of rectangle with perimeter 20 maximizing area yields a 5x5 square", "Minimizing f(x) = x^3 - 3x on [0, 2] requires checking critical points and endpoints", "Optimizing f(x) = x^2 - 4x + 5 gives minimum at x = 2"],
        "mistakes": ["Forgetting to check endpoints on closed intervals", "Assuming critical points are always extrema (could be inflection)", "Confusing local and global extrema", "Forgetting that f' doesn't exist can also be a critical point", "Errors in second derivative sign interpretation"],
    },
    "taylor_series": {
        "def": "the representation of a function as an infinite power series f(x) = sum f^(n)(a)/n! * (x-a)^n around a point a, enabling polynomial approximations of smooth functions",
        "principles": ["Taylor series at a: sum_{n=0}^inf f^(n)(a)/n! (x-a)^n", "Maclaurin series is Taylor series at a = 0", "Common series: e^x = sum x^n/n!, sin(x) = sum (-1)^n x^(2n+1)/(2n+1)!, cos(x) = sum (-1)^n x^(2n)/(2n)!", "Radius of convergence determined by ratio test", "Truncated Taylor polynomials approximate the function near a"],
        "examples": ["Maclaurin series of e^x = 1 + x + x^2/2! + x^3/3! + ...", "Taylor series of ln(x) at x = 1: (x-1) - (x-1)^2/2 + (x-1)^3/3 - ...", "Approximating sin(0.1) using first three terms of its series"],
        "mistakes": ["Confusing Taylor (at any a) with Maclaurin (at a = 0)", "Forgetting factorial denominators", "Assuming all functions have a Taylor series (need smoothness)", "Errors in computing higher derivatives", "Confusing radius of convergence with interval of convergence"],
    },
    "integration_techniques": {
        "def": "methods for evaluating antiderivatives including substitution (reverse chain rule), integration by parts (reverse product rule), partial fractions, trigonometric substitution, and tabular integration",
        "principles": ["u-substitution: integral f(g(x)) g'(x) dx = integral f(u) du", "Integration by parts: integral u dv = uv - integral v du", "Partial fractions decompose rational functions into simpler terms", "Trig substitution handles integrals with sqrt(a^2 - x^2), etc.", "Some integrals have no elementary antiderivative"],
        "examples": ["integral 2x cos(x^2) dx = sin(x^2) + C by u-substitution", "integral x e^x dx = x e^x - e^x + C by integration by parts", "integral 1/(x^2 - 1) dx = (1/2) ln|x-1| - (1/2) ln|x+1| + C by partial fractions"],
        "mistakes": ["Forgetting to change dx to du in substitution", "Choosing u and dv poorly in integration by parts", "Errors in partial fraction decomposition setup", "Forgetting the constant of integration", "Misapplying trig substitution identities"],
    },
    "definite_integrals": {
        "def": "integrals with bounds computing the signed area between a function and the x-axis over an interval, evaluated via the Fundamental Theorem of Calculus as F(b) - F(a)",
        "principles": ["Fundamental Theorem: integral_a^b f(x) dx = F(b) - F(a) where F' = f", "Definite integrals can be negative if function is below x-axis", "Properties: integral_a^b f + integral_b^c f = integral_a^c f", "Symmetry: integral_-a^a f(x) dx = 0 if f is odd, 2 * integral_0^a f if f is even", "Substitution requires changing bounds when used in definite integrals"],
        "examples": ["integral_0^1 x^2 dx = 1/3", "integral_0^pi sin(x) dx = 2", "integral_-1^1 x^3 dx = 0 by symmetry (odd function)"],
        "mistakes": ["Forgetting to evaluate at bounds vs leaving + C", "Not changing bounds when doing u-substitution", "Confusing definite integral value with antiderivative", "Forgetting that integrals can be negative", "Sign errors when function is below x-axis"],
    },
    "improper_integrals": {
        "def": "integrals over infinite intervals or with unbounded integrands, evaluated as limits of proper integrals, possibly convergent or divergent",
        "principles": ["Type 1: infinite interval; integral_a^inf f = lim_{b->inf} integral_a^b f", "Type 2: unbounded integrand; integral_a^b f with singularity at a or b evaluated as a limit", "p-test: integral_1^inf 1/x^p converges iff p > 1", "Comparison tests establish convergence/divergence", "Some improper integrals converge despite infinite domains"],
        "examples": ["integral_1^inf 1/x^2 dx = 1 (converges)", "integral_1^inf 1/x dx diverges (harmonic series analog)", "integral_0^1 1/sqrt(x) dx = 2 (converges despite singularity at 0)"],
        "mistakes": ["Treating improper integrals as ordinary integrals without limits", "Forgetting to check for singularities inside the interval", "Misapplying p-test (p > 1 for convergence at infinity)", "Errors in computing limits", "Assuming infinite domains always mean divergence"],
    },
    "multivariable_calculus": {
        "def": "calculus of functions of several variables, including partial derivatives, gradients, multiple integrals, and vector calculus, generalizing single-variable calculus to higher dimensions",
        "principles": ["Partial derivatives differentiate with respect to one variable holding others constant", "Gradient is the vector of partial derivatives", "Directional derivative = grad f . unit vector", "Chain rule generalizes to multiple paths", "Double and triple integrals compute over 2D and 3D regions"],
        "examples": ["f(x,y) = x^2 y^3 has partial derivatives 2xy^3 and 3x^2 y^2", "Gradient of f(x,y) = x^2 + y^2 is <2x, 2y>", "Double integral over a rectangle computed as iterated integral"],
        "mistakes": ["Confusing partial derivatives with total derivatives", "Errors in iterated integral bounds", "Forgetting to convert dA to dx dy or polar form", "Confusing gradient with directional derivative", "Not checking continuity before differentiating"],
    },
    "partial_derivatives": {
        "def": "derivatives of multivariable functions with respect to one variable while holding others constant, generalizing ordinary derivatives to functions of several variables",
        "principles": ["Partial f / partial x holds y, z, ... constant", "Clairaut's theorem: mixed partials are equal if continuous", "Higher-order partials: f_xx, f_xy, f_yy, etc.", "Partial derivatives commute when continuous", "Total derivative accounts for all variable changes"],
        "examples": ["f(x,y) = x^2 y; partial f/partial x = 2xy, partial f/partial y = x^2", "f(x,y,z) = xyz; partial f/partial z = xy", "Verifying Clairaut: f_xy = f_yx for f = x^2 y^3"],
        "mistakes": ["Forgetting to hold other variables constant", "Errors in higher-order mixed partials", "Confusing partial with total derivative", "Assuming mixed partials are always equal (need continuity)", "Errors in chain rule for multivariable functions"],
    },
    "lagrange_multipliers": {
        "def": "a method for finding extrema of a function subject to equality constraints by introducing auxiliary variables (multipliers) and solving the gradient equations grad f = lambda grad g",
        "principles": ["At constrained extremum: grad f = lambda grad g", "Multiple constraints: grad f = sum lambda_i grad g_i", "Solve system of gradient equations and constraints simultaneously", "Method finds candidates; classify separately", "Geometric: gradients are parallel at extremum"],
        "examples": ["Maximize f(x,y) = xy subject to x + y = 10 yields x = y = 5", "Find closest point on plane ax+by+cz=d to origin", "Minimize x^2 + y^2 subject to xy = 1"],
        "mistakes": ["Forgetting to also solve the constraint equation", "Confusing the multiplier's role", "Errors in solving the system of equations", "Not checking boundary or endpoints separately", "Misclassifying extrema found via Lagrange"],
    },
    "vector_calculus": {
        "def": "calculus of vector fields including gradient, divergence, curl, line integrals, surface integrals, and the theorems of Green, Gauss (divergence), and Stokes",
        "principles": ["Gradient of scalar field is a vector field", "Divergence of vector field is a scalar measuring source/sink", "Curl of vector field is a vector measuring rotation", "Line integral integrates a field along a curve", "Green's, Stokes', and Divergence theorems relate integral types"],
        "examples": ["Gradient of f(x,y,z) = x^2 + y^2 + z^2 is <2x, 2y, 2z>", "Divergence of F = <x, y, z> is 3", "Curl of F = <-y, x, 0> is <0, 0, 2>"],
        "mistakes": ["Confusing gradient, divergence, and curl", "Errors in line integral parameterization", "Forgetting orientation in surface integrals", "Misapplying Green's, Stokes', or Divergence theorems", "Confusing scalar and vector fields"],
    },
    "line_integrals": {
        "def": "integrals of scalar or vector fields along a curve, generalizing ordinary integrals to integration over paths in 2D or 3D space",
        "principles": ["Scalar line integral: integral_C f(x,y,z) ds", "Vector line integral: integral_C F . dr = integral_C F(r(t)) . r'(t) dt", "Parameterize curve r(t), a <= t <= b, then integrate", "Path independence: equivalent to conservative field for vector line integrals", "Fundamental theorem for line integrals: integral grad f . dr = f(end) - f(start)"],
        "examples": ["integral_C (x + y) ds along r(t) = (t, t), 0 <= t <= 1", "Work done by force F along path C = integral_C F . dr", "Verifying path independence for conservative fields"],
        "mistakes": ["Forgetting to compute r'(t) for vector line integrals", "Errors in curve parameterization", "Confusing scalar and vector line integrals", "Not checking if field is conservative for path independence", "Forgetting |r'(t)| factor for scalar line integrals"],
    },
    "greens_theorem": {
        "def": "a theorem relating a line integral around a simple closed curve C to a double integral over the region D bounded by C: integral_C P dx + Q dy = double integral_D (partial Q/partial x - partial P/partial y) dA",
        "principles": ["Relates line integral around boundary to area integral of curl", "Requires positively oriented, piecewise smooth, simple closed curve", "Useful for converting line integrals to area integrals and vice versa", "Generalizes Fundamental Theorem of Calculus to 2D", "Special case of Stokes' theorem"],
        "examples": ["Computing line integral via double integral using Green's theorem", "Computing area of region via (1/2) integral_C (x dy - y dx)", "Verifying Green's theorem on a simple example"],
        "mistakes": ["Forgetting orientation (positive = counterclockwise)", "Errors in identifying P and Q", "Misapplying to non-simple or non-closed curves", "Confusing with Stokes' or Divergence theorems", "Sign errors in the formula"],
    },
    "vectors": {
        "def": "mathematical objects with magnitude and direction, represented as ordered tuples of components, with operations including addition, scalar multiplication, dot product, and cross product",
        "principles": ["Vector addition is component-wise", "Scalar multiplication scales each component", "Dot product: u . v = sum u_i v_i = |u||v| cos(theta)", "Cross product (3D): u x v is perpendicular to both, magnitude |u||v| sin(theta)", "Unit vectors have length 1; normalize by dividing by magnitude"],
        "examples": ["<1, 2, 3> + <4, 5, 6> = <5, 7, 9>", "<1, 0, 0> . <0, 1, 0> = 0 (orthogonal)", "<1, 0, 0> x <0, 1, 0> = <0, 0, 1>"],
        "mistakes": ["Confusing dot product with cross product", "Forgetting that cross product is anticommutative: u x v = -(v x u)", "Treating vectors as scalars in operations", "Errors in computing magnitudes", "Confusing row and column vectors in matrix context"],
    },
    "matrices": {
        "def": "rectangular arrays of numbers with operations including addition, scalar multiplication, matrix multiplication, transpose, and inverse, used to represent linear transformations",
        "principles": ["Matrix addition is element-wise (same dimensions)", "Matrix multiplication: (AB)_{ij} = sum_k A_{ik} B_{kj}", "Multiplication is not commutative: AB != BA in general", "Transpose A^T swaps rows and columns", "Inverse A^(-1) satisfies A A^(-1) = I if A is invertible"],
        "examples": ["[[1,2],[3,4]] + [[5,6],[7,8]] = [[6,8],[10,12]]", "[[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]", "Transpose of [[1,2,3]] is [[1],[2],[3]]"],
        "mistakes": ["Treating matrix multiplication as element-wise", "Assuming AB = BA", "Errors in tracking indices in multiplication", "Forgetting that not all matrices have inverses", "Confusing row and column conventions"],
    },
    "determinants": {
        "def": "a scalar value associated with a square matrix, computed via cofactor expansion or row reduction, indicating invertibility (nonzero determinant) and geometric scaling factor",
        "principles": ["det(A) nonzero iff A is invertible", "det(AB) = det(A) det(B)", "det(A^T) = det(A)", "Row operations affect determinant predictably (swapping rows negates)", "Geometrically |det| is the volume scaling factor of the linear transformation"],
        "examples": ["det([[a,b],[c,d]]) = ad - bc for 2x2", "Cofactor expansion for 3x3 matrices", "Triangular matrix determinant is the product of diagonal entries"],
        "mistakes": ["Errors in cofactor expansion signs", "Forgetting that row swaps negate the determinant", "Confusing determinant with trace", "Assuming det(A + B) = det(A) + det(B) (generally false)", "Errors in computing larger determinants"],
    },
    "eigenvalues_eigenvectors": {
        "def": "for a square matrix A, scalars lambda and nonzero vectors v satisfying Av = lambda v, with eigenvalues found via the characteristic polynomial det(A - lambda I) = 0",
        "principles": ["Av = lambda v defines eigenvalue-eigenvector pair", "Characteristic polynomial: det(A - lambda I) = 0", "Trace = sum of eigenvalues; determinant = product of eigenvalues", "Diagonalizable matrices have a basis of eigenvectors", "Symmetric matrices have real eigenvalues and orthogonal eigenvectors"],
        "examples": ["Eigenvalues of [[2,0],[0,3]] are 2 and 3", "Finding eigenvalues of [[1,2],[2,1]] yields 3 and -1", "Symmetric matrix [[2,1],[1,2]] has eigenvalues 3 and 1 with orthogonal eigenvectors"],
        "mistakes": ["Errors in computing the characteristic polynomial", "Confusing eigenvalues with diagonal entries", "Forgetting that not all matrices are diagonalizable", "Not normalizing eigenvectors when required", "Assuming eigenvectors are unique (they're determined up to scale)"],
    },
    "svd": {
        "def": "the Singular Value Decomposition factorizing any m x n matrix A as A = U Sigma V^T, where U and V are orthogonal and Sigma is diagonal with non-negative singular values",
        "principles": ["Every matrix has an SVD, even non-square ones", "Singular values are the square roots of eigenvalues of A^T A", "Columns of U are left singular vectors; columns of V are right singular vectors", "Rank equals number of nonzero singular values", "Truncated SVD gives best low-rank approximation"],
        "examples": ["SVD of a 2x2 matrix yielding U, Sigma, V^T", "Truncated SVD for image compression", "PCA as SVD on centered data matrix"],
        "mistakes": ["Confusing singular values with eigenvalues", "Forgetting that singular values are non-negative", "Errors in computing U and V from eigenvectors", "Confusing left and right singular vectors", "Assuming SVD is unique — it's unique up to signs and ordering"],
    },
    "vector_spaces": {
        "def": "algebraic structures consisting of vectors that can be added and scaled, satisfying axioms like associativity, distributivity, and existence of zero and inverses",
        "principles": ["Closure under addition and scalar multiplication", "Eight axioms: associativity, commutativity, identity, inverse, distributivity (2), compatibility", "Subspaces are subsets closed under addition and scalar multiplication", "Span of vectors is the set of all linear combinations", "Linear independence: no vector is a combination of others"],
        "examples": ["R^n is a vector space over R", "Polynomials of degree <= n form a vector space", "Continuous functions on [0,1] form a vector space"],
        "mistakes": ["Confusing subspaces with subsets (need closure)", "Errors in checking linear independence", "Forgetting to verify all eight axioms for unusual spaces", "Confusing span with basis", "Treating vectors as ordered tuples in abstract spaces"],
    },
    "linear_transformations": {
        "def": "functions between vector spaces that preserve vector addition and scalar multiplication, representable as matrices once bases are chosen",
        "principles": ["T(u + v) = T(u) + T(v); T(cu) = c T(u)", "Every linear transformation can be represented by a matrix", "Kernel is the set mapped to zero; image is the range", "Rank-nullity theorem: dim(domain) = rank + nullity", "Composition corresponds to matrix multiplication"],
        "examples": ["T(x, y) = (2x, x + y) is linear; matrix is [[2,0],[1,1]]", "Rotation by angle theta has matrix [[cos,-sin],[sin,cos]]", "Projection onto x-axis: T(x,y) = (x, 0)"],
        "mistakes": ["Confusing linear with affine transformations", "Forgetting that translations are not linear (they're affine)", "Errors in computing kernel and image", "Misapplying rank-nullity theorem", "Confusing the matrix representation with the transformation itself"],
    },
    "inner_product_spaces": {
        "def": "vector spaces equipped with an inner product satisfying positivity, symmetry (or conjugate symmetry), and linearity, enabling notions of length, angle, and orthogonality",
        "principles": ["Inner product <u, v> generalizes dot product", "Norm: ||v|| = sqrt(<v, v>)", "Cauchy-Schwarz: |<u, v>| <= ||u|| ||v||", "Triangle inequality: ||u + v|| <= ||u|| + ||v||", "Orthogonality: <u, v> = 0; Pythagorean theorem generalizes"],
        "examples": ["Standard dot product in R^n", "Inner product <f, g> = integral f g on function spaces", "Weighted inner products <u, v>_A = u^T A v"],
        "mistakes": ["Confusing inner product with norm", "Forgetting conjugate symmetry in complex spaces", "Misapplying Cauchy-Schwarz", "Errors in computing norms in function spaces", "Assuming all inner products are the dot product"],
    },
    "probability_foundations": {
        "def": "the mathematical framework for quantifying uncertainty, built on sample spaces, events, probability axioms (Kolmogorov), and the rules of combinatorial and conditional probability",
        "principles": ["Probability axioms: non-negative, total = 1, countable additivity for disjoint events", "Complement rule: P(A^c) = 1 - P(A)", "Addition rule: P(A or B) = P(A) + P(B) - P(A and B)", "Conditional probability: P(A | B) = P(A and B) / P(B) if P(B) > 0", "Independence: A and B independent iff P(A and B) = P(A)P(B)"],
        "examples": ["Rolling a fair die: P(even) = 3/6 = 1/2", "Drawing a card: P(king or heart) = 4/52 + 13/52 - 1/52 = 16/52", "P(rain | clouds) generally higher than P(rain)"],
        "mistakes": ["Treating non-disjoint events as disjoint (forgetting intersection)", "Confusing independence with mutual exclusivity", "Errors in applying conditional probability", "Forgetting that P(A | B) != P(B | A) in general", "Not normalizing probabilities to sum to 1"],
    },
    "conditional_probability": {
        "def": "the probability of an event given that another has occurred, defined as P(A | B) = P(A and B) / P(B), foundational to Bayes' theorem and dependence reasoning",
        "principles": ["P(A | B) = P(A and B) / P(B) requires P(B) > 0", "If A and B independent: P(A | B) = P(A)", "Chain rule: P(A and B) = P(A | B) P(B)", "Law of total probability: P(A) = sum P(A | B_i) P(B_i) for partition", "Conditional probabilities can be counterintuitive"],
        "examples": ["P(disease | positive test) computed via Bayes' theorem", "P(rain | humid) typically higher than P(rain)", "Monty Hall problem showcasing surprising conditional probability"],
        "mistakes": ["Confusing P(A | B) with P(B | A) (prosecutor's fallacy)", "Forgetting to check P(B) > 0", "Errors in applying law of total probability", "Treating dependent events as independent", "Not partitioning correctly in total probability"],
    },
    "bayes_theorem": {
        "def": "the formula P(A | B) = P(B | A) P(A) / P(B) relating conditional probabilities, enabling updating of beliefs with evidence and foundational to Bayesian statistics",
        "principles": ["P(A | B) = P(B | A) P(A) / P(B)", "P(A) is the prior; P(A | B) is the posterior", "P(B | A) is the likelihood; P(B) is the evidence", "Generalizes to multiple hypotheses via law of total probability", "Bayesian updating revises prior with new evidence"],
        "examples": ["Medical testing: computing P(disease | positive) accounting for false positive rate", "Spam filtering: P(spam | words) using word frequencies", "Updating belief in a hypothesis as new data arrives"],
        "mistakes": ["Confusing P(A | B) with P(B | A) (base rate neglect)", "Forgetting to compute P(B) via total probability", "Misinterpreting prior as fixed rather than updatable", "Errors in identifying likelihood and evidence", "Treating small probabilities as zero"],
    },
    "random_variables": {
        "def": "functions mapping outcomes of a random experiment to real numbers, characterized by their probability distributions (PMF for discrete, PDF for continuous) and moments like expectation and variance",
        "principles": ["Discrete RVs have PMF: P(X = x)", "Continuous RVs have PDF f(x) with integral = 1", "CDF F(x) = P(X <= x) for both types", "Expectation E[X] = sum x P(X=x) or integral x f(x) dx", "Variance Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2"],
        "examples": ["Rolling a die: X takes values 1-6 with P=1/6 each", "Exponential RV with PDF f(x) = lambda e^(-lambda x)", "Computing E[X] and Var(X) for a Bernoulli(p)"],
        "mistakes": ["Confusing PMF with PDF", "Forgetting to verify PDFs integrate to 1", "Errors in computing expectations", "Confusing variance with standard deviation", "Forgetting that E[X^2] != (E[X])^2 in general"],
    },
    "distributions_discrete": {
        "def": "probability distributions over countable outcomes including Bernoulli, Binomial, Geometric, Poisson, and Negative Binomial, each modeling specific counting or waiting scenarios",
        "principles": ["Bernoulli(p): single trial, P(X=1) = p", "Binomial(n, p): number of successes in n independent trials", "Geometric(p): trials until first success", "Poisson(lambda): number of rare events in fixed interval", "Expectation and variance formulas vary by distribution"],
        "examples": ["Coin flip: Bernoulli(0.5)", "20 coin flips: Binomial(20, 0.5)", "Customer arrivals per hour often modeled as Poisson(5)"],
        "mistakes": ["Confusing Binomial with Poisson", "Forgetting independence requirement for Binomial", "Errors in mean and variance formulas", "Misapplying Geometric to non-iid sequences", "Confusing PDF with CDF for discrete distributions"],
    },
    "distributions_continuous": {
        "def": "probability distributions over continuous ranges including Uniform, Normal (Gaussian), Exponential, Gamma, and Beta, characterized by their PDFs and used to model continuous phenomena",
        "principles": ["Normal(mu, sigma^2): bell curve with mean mu, variance sigma^2", "Uniform(a, b): constant PDF on [a, b]", "Exponential(lambda): time between Poisson events", "Gamma and Beta are flexible families for positive and bounded variables", "Standardization: Z = (X - mu)/sigma is standard Normal"],
        "examples": ["Heights often modeled as Normal(170, 100)", "Wait times as Exponential(0.5) per minute", "Uniform(0, 1) for generating random numbers"],
        "mistakes": ["Confusing PDF value with probability (PDF can exceed 1)", "Forgetting parameters of Normal: mean and variance (not std dev)", "Errors in CDF computations", "Misidentifying when to use Exponential vs Gamma", "Confusing standard Normal with general Normal"],
    },
    "descriptive_statistics": {
        "def": "methods for summarizing and describing data including measures of center (mean, median, mode), spread (variance, std dev, IQR), shape (skewness, kurtosis), and visualizations (histograms, box plots)",
        "principles": ["Mean is sensitive to outliers; median is robust", "Variance = average squared deviation; std dev = sqrt(variance)", "IQR = Q3 - Q1 measures robust spread", "Skewness measures asymmetry; kurtosis measures tail weight", "Five-number summary: min, Q1, median, Q3, max"],
        "examples": ["Computing mean and median for [1, 2, 3, 4, 100]", "Box plot showing IQR and outliers", "Histogram revealing bimodal distribution"],
        "mistakes": ["Confusing sample and population formulas for variance", "Forgetting to handle outliers when interpreting mean", "Confusing skewness direction (right-skew has long right tail)", "Errors in computing quartiles", "Misinterpreting standard deviation as average absolute deviation"],
    },
    "hypothesis_testing": {
        "def": "a framework for making decisions about population parameters using sample data, involving null and alternative hypotheses, test statistics, p-values, significance levels, and Type I/II errors",
        "principles": ["Null hypothesis H0 typically represents no effect or status quo", "Alternative H1 represents the claim being tested", "p-value: probability of observing data as extreme under H0", "Significance level alpha is the threshold for rejecting H0", "Type I error: reject true H0; Type II error: fail to reject false H0"],
        "examples": ["Testing if a coin is fair: H0: p = 0.5 vs H1: p != 0.5", "Two-sample t-test comparing means of two groups", "Chi-square test of independence for categorical variables"],
        "mistakes": ["Confusing p-value with probability that H0 is true", "Equating statistical significance with practical significance", "Multiple testing without correction inflating Type I error", "Misinterpreting failure to reject H0 as evidence for H0", "Errors in choosing appropriate test for data type"],
    },
}
SEEDS.update(MATH_SEEDS)

# Sanity check seeds
for domain, topics in SUBTOPICS.items():
    for t in topics:
        assert t in SEEDS, f"missing seed for {domain}/{t}"
print(f"[ok] {len(SEEDS)} seeds loaded across {len(SUBTOPICS)} domains")

# ---------------------------------------------------------------------------
# PROMPT TEMPLATES — 15 per domain (60 total)
# ---------------------------------------------------------------------------
TEMPLATES = {
    "applied_neuroscience": [
        "Explain {topic} for {audience} {perspective}.",
        "What is {topic}? Provide a {format}.",
        "Discuss {topic} {perspective}, addressing {audience}.",
        "Define {topic} and outline its core principles, then illustrate with a {format}.",
        "Why does {topic} matter in neuroscience? Answer as if for {audience}.",
        "Walk me through {topic} using a {format}, {perspective}.",
        "Teach {topic} to {audience}, {perspective}.",
        "Critically evaluate {topic} {perspective}. Include a {format}.",
        "What are the foundations of {topic}? Provide a {format} {perspective}.",
        "Compare and contrast views on {topic}, addressing {audience}.",
        "Describe {topic} with a {format} that would suit {audience}.",
        "How is {topic} studied experimentally? Answer {perspective}.",
        "What are common misconceptions about {topic}? Provide a {format}.",
        "Connect {topic} to broader brain function {perspective}.",
        "Summarize {topic} {perspective} for {audience}.",
    ],
    "rust": [
        "Explain {topic} in Rust for {audience}, {perspective}.",
        "What is {topic} in Rust? Provide a {format} with code.",
        "Discuss {topic} {perspective}, addressing {audience}.",
        "Define {topic} and show its core usage, then illustrate with a {format}.",
        "Why does {topic} matter in Rust? Answer as if for {audience}.",
        "Walk me through {topic} using a {format}, {perspective}.",
        "Teach {topic} to {audience}, {perspective}.",
        "Critically evaluate {topic} {perspective}. Include a {format}.",
        "What are the foundations of {topic}? Provide a {format} {perspective}.",
        "Compare and contrast approaches to {topic}, addressing {audience}.",
        "Describe {topic} with a {format} that would suit {audience}.",
        "How does {topic} interact with the borrow checker? Answer {perspective}.",
        "What are common mistakes with {topic}? Provide a {format}.",
        "Connect {topic} to idiomatic Rust {perspective}.",
        "Summarize {topic} {perspective} for {audience}.",
    ],
    "python": [
        "Explain {topic} in Python for {audience}, {perspective}.",
        "What is {topic} in Python? Provide a {format} with code.",
        "Discuss {topic} {perspective}, addressing {audience}.",
        "Define {topic} and show its core usage, then illustrate with a {format}.",
        "Why does {topic} matter in Python? Answer as if for {audience}.",
        "Walk me through {topic} using a {format}, {perspective}.",
        "Teach {topic} to {audience}, {perspective}.",
        "Critically evaluate {topic} {perspective}. Include a {format}.",
        "What are the foundations of {topic}? Provide a {format} {perspective}.",
        "Compare and contrast approaches to {topic}, addressing {audience}.",
        "Describe {topic} with a {format} that would suit {audience}.",
        "How does {topic} interact with Python's object model? Answer {perspective}.",
        "What are common mistakes with {topic}? Provide a {format}.",
        "Connect {topic} to idiomatic Python {perspective}.",
        "Summarize {topic} {perspective} for {audience}.",
    ],
    "math": [
        "Explain {topic} for {audience}, {perspective}.",
        "What is {topic} in mathematics? Provide a {format}.",
        "Discuss {topic} {perspective}, addressing {audience}.",
        "Define {topic} and outline its core principles, then illustrate with a {format}.",
        "Why does {topic} matter in mathematics? Answer as if for {audience}.",
        "Walk me through {topic} using a {format}, {perspective}.",
        "Teach {topic} to {audience}, {perspective}.",
        "Critically evaluate {topic} {perspective}. Include a {format}.",
        "What are the foundations of {topic}? Provide a {format} {perspective}.",
        "Compare and contrast approaches to {topic}, addressing {audience}.",
        "Describe {topic} with a {format} that would suit {audience}.",
        "How is {topic} proved or derived? Answer {perspective}.",
        "What are common mistakes with {topic}? Provide a {format}.",
        "Connect {topic} to broader mathematical theory {perspective}.",
        "Summarize {topic} {perspective} for {audience}.",
    ],
}


# ---------------------------------------------------------------------------
# ASSISTANT RESPONSE BUILDER
# ---------------------------------------------------------------------------
# Domain-specific openers/prefixes that introduce variation
DOMAIN_OPENERS = {
    "applied_neuroscience": [
        "From a neuroscience perspective,",
        "In the context of neural computation,",
        "Looking at this through the lens of brain function,",
        "From the standpoint of systems neuroscience,",
        "Approaching this question as a neuroscientist,",
    ],
    "rust": [
        "In Rust,",
        "From a Rust programmer's perspective,",
        "Looking at this through Rust's ownership model,",
        "From the standpoint of safe systems programming,",
        "Approaching this in idiomatic Rust,",
    ],
    "python": [
        "In Python,",
        "From a Pythonista's perspective,",
        "Looking at this through Python's object model,",
        "From the standpoint of dynamic language design,",
        "Approaching this in idiomatic Python,",
    ],
    "math": [
        "From a mathematical standpoint,",
        "In mathematical terms,",
        "Looking at this through the lens of pure mathematics,",
        "From the perspective of mathematical structure,",
        "Approaching this rigorously,",
    ],
}


def build_assistant(domain, subtopic, seed, audience, perspective, fmt, opener, principles_ord=None, examples_ord=None, mistakes_ord=None):
    """Build a long, structured assistant response (500-900 tokens target).

    principles_ord / examples_ord / mistakes_ord are optional permutations
    of indices into the corresponding seed lists, enabling combinatorial
    variation in the assistant response independent of the user prompt.
    """
    topic_pretty = subtopic.replace("_", " ")
    # Resolve permuted lists (fall back to identity if not provided)
    if principles_ord is None:
        principles_ord = list(range(len(seed["principles"])))
    if examples_ord is None:
        examples_ord = list(range(len(seed["examples"])))
    if mistakes_ord is None:
        mistakes_ord = list(range(len(seed["mistakes"])))
    principles = [seed["principles"][i] for i in principles_ord]
    examples = [seed["examples"][i] for i in examples_ord]
    mistakes = [seed["mistakes"][i] for i in mistakes_ord]
    # Compose response in clearly delineated sections for richness.
    parts = []
    parts.append(f"{opener} **{topic_pretty}** is {seed['def']}.")

    # Principles section
    parts.append("\n\n**Core principles:**")
    for i, p in enumerate(principles, 1):
        parts.append(f"\n{i}. {p}")

    # Connect to the requested perspective
    parts.append(f"\n\nLooking at this {perspective}, the key insight is that {principles[0].lower()} — this anchors the rest of the discussion.")

    # Format-specific block
    if fmt == "step-by-step procedure":
        parts.append(f"\n\n**Step-by-step procedure:**")
        parts.append(f"\n1. **Identify the goal.** When working with {topic_pretty}, the first task is to clarify what you are trying to achieve {perspective}.")
        parts.append(f"\n2. **Recall the definition.** Remind yourself that {topic_pretty} is {seed['def']}")
        parts.append(f"\n3. **Apply the principles.** Use the principles listed above as a checklist. The first principle — {principles[0].lower()} — is especially important to keep in mind.")
        parts.append(f"\n4. **Work an example.** Take one of the examples below and walk through it carefully, checking each step against the principles.")
        parts.append(f"\n5. **Check for mistakes.** Compare your work against the common mistakes listed to ensure you have not fallen into a trap.")
        parts.append(f"\n6. **Generalize.** Once the example feels solid, ask how {topic_pretty} connects to related concepts in the broader field.")
    elif fmt == "bulleted checklist":
        parts.append(f"\n\n**Checklist for working with {topic_pretty}:**")
        for p in principles:
            parts.append(f"\n- [ ] {p}")
        parts.append(f"\n- [ ] Cross-check against the definition: {seed['def']}")
        parts.append(f"\n- [ ] Identify at least one concrete example")
        parts.append(f"\n- [ ] Identify at least one common mistake to avoid")
    elif fmt == "worked example with concrete numbers":
        parts.append(f"\n\n**Worked example:**")
        parts.append(f"\nConsider this scenario: {examples[0]}.")
        parts.append(f"\n1. We start with the definition: {topic_pretty} is {seed['def']}")
        parts.append(f"\n2. Apply principle: {principles[0]}.")
        parts.append(f"\n3. Apply principle: {principles[1]}.")
        parts.append(f"\n4. The example illustrates how {principles[2].lower()} plays out in practice.")
        parts.append(f"\n5. Verification: ensure the result is consistent with the definition and the principles, and that none of the common mistakes apply.")
    elif fmt == "scenario walkthrough":
        parts.append(f"\n\n**Scenario walkthrough:**")
        parts.append(f"\nImagine you are {audience}. You encounter {topic_pretty} for the first time.")
        parts.append(f"\n- You read that {topic_pretty} is {seed['def']}")
        parts.append(f"\n- You ask: what makes this concept important? The answer is that {principles[0].lower()}.")
        parts.append(f"\n- You try the example: {examples[0]}. This concretizes the idea.")
        parts.append(f"\n- You stumble on a common mistake: {mistakes[0]}. Recognizing this sharpens your understanding.")
        parts.append(f"\n- Finally, you connect {topic_pretty} to related ideas {perspective}.")
    elif fmt == "Q&A dialogue":
        parts.append(f"\n\n**Q&A dialogue:**")
        parts.append(f"\n**Q:** What is {topic_pretty}?")
        parts.append(f"\n**A:** {seed['def'].capitalize()}.")
        parts.append(f"\n**Q:** What are the core principles?")
        parts.append(f"\n**A:** The core principles are: {', '.join(p.lower() for p in principles)}.")
        parts.append(f"\n**Q:** Can you give an example?")
        parts.append(f"\n**A:** Sure — {examples[0]}.")
        parts.append(f"\n**Q:** What mistakes should I avoid?")
        parts.append(f"\n**A:** A common one is {mistakes[0].lower()}")
    elif fmt == "compare-and-contrast table description":
        parts.append(f"\n\n**Compare-and-contrast:**")
        parts.append(f"\n| Aspect | {topic_pretty} | Related concept |")
        parts.append(f"\n|---|---|---|")
        parts.append(f"\n| Definition | {seed['def']} | A related but distinct idea |")
        parts.append(f"\n| Key principle | {principles[0]} | Different principle |")
        parts.append(f"\n| Example | {examples[0]} | Different example |")
        parts.append(f"\n| Common mistake | {mistakes[0]} | Different mistake |")
    elif fmt == "decision tree narrative":
        parts.append(f"\n\n**Decision tree:**")
        parts.append(f"\n- If you are {audience}: start with the definition: {seed['def']}")
        parts.append(f"\n  - If you understand the definition: proceed to the principles, starting with {principles[0].lower()}")
        parts.append(f"\n    - If the principles feel clear: try the example: {examples[0]}")
        parts.append(f"\n      - If the example works: check for mistakes, especially {mistakes[0].lower()}")
        parts.append(f"\n      - If the example fails: revisit the principles and the definition.")
        parts.append(f"\n    - If the principles feel unclear: re-read the definition and try a simpler perspective.")
        parts.append(f"\n  - If the definition is unclear: try framing it {perspective}.")
    elif fmt == "common mistakes and corrections":
        parts.append(f"\n\n**Common mistakes and corrections:**")
        for i, m in enumerate(mistakes, 1):
            parts.append(f"\n{i}. **Mistake:** {m}")
            parts.append(f"\n   **Correction:** Re-anchor on the principle that {principles[i % len(principles)].lower()}. Re-read the definition: {seed['def']}")
    elif fmt == "before/after transformation":
        parts.append(f"\n\n**Before/after transformation:**")
        parts.append(f"\n**Before understanding:** A naive view of {topic_pretty} might be vague or incomplete.")
        parts.append(f"\n**After understanding:** With the definition — {seed['def']} — and the principles in mind, the concept becomes precise and actionable.")
        parts.append(f"\n**Key shift:** The biggest shift is recognizing that {principles[0].lower()}.")
        parts.append(f"\n**Verification:** Check by working through {examples[0]}.")
    elif fmt == "principles-then-application":
        parts.append(f"\n\n**Principles, then application:**")
        parts.append(f"\n*Principles:*")
        for i, p in enumerate(principles, 1):
            parts.append(f"\n{i}. {p}")
        parts.append(f"\n\n*Application:*")
        parts.append(f"\nTake the example: {examples[0]}.")
        parts.append(f"\n- Apply principle 1: {principles[0]} → observe how it constrains the situation.")
        parts.append(f"\n- Apply principle 2: {principles[1]} → see how it guides the next step.")
        parts.append(f"\n- Apply principle 3: {principles[2]} → confirm the result.")
        parts.append(f"\n- Cross-check against common mistakes: avoid {mistakes[0].lower()}.")

    # Examples section
    parts.append(f"\n\n**Concrete examples:**")
    for i, ex in enumerate(examples, 1):
        parts.append(f"\n{i}. {ex}")

    # Common mistakes section
    parts.append(f"\n\n**Common mistakes to avoid:**")
    for i, m in enumerate(mistakes, 1):
        parts.append(f"\n{i}. {m}")

    # Audience-specific closing
    parts.append(f"\n\n**For {audience} specifically:**")
    parts.append(f"\n{audience.capitalize()} should focus on the definition first, then carefully apply principle: {principles[0].lower()}. Once that foundation is in place, work through the examples and check against the common mistakes. The perspective — {perspective} — offers a useful lens for deepening understanding, but the definition and principles are the anchor.")

    # Final synthesis
    parts.append(f"\n\n**Synthesis:** {topic_pretty} connects definition, principles, examples, and mistakes into a coherent concept. The definition sets the scope; the principles constrain the application; the examples ground the abstraction; and the mistakes highlight the pitfalls. Approaching the concept {perspective}, while keeping {audience} in mind, makes the abstract concrete and the concrete abstract.")

    return "".join(parts)


def _perm_from_seed(n, seed_int):
    """Deterministically permute list(range(n)) using a seed int."""
    items = list(range(n))
    rng = random.Random(seed_int)
    rng.shuffle(items)
    return items


def build_entry(domain, subtopic, idx):
    """Build a single JSONL entry deterministically from idx.

    Variation matrix (per subtopic):
      - 15 templates x 10 audiences x 10 perspectives x 10 formats = 15,000 user strings
      - 5 openers x 5! principles x 3! examples x 5! mistakes = 86,400 assistant variants
      - Combined combinatorial space per subtopic >> 7,500 target entries.
    Dedup happens in write_file on FULL entry hash (user+assistant).
    """
    seed = SEEDS[subtopic]
    topic_pretty = subtopic.replace("_", " ")
    n_tpl = len(TEMPLATES[domain])
    n_aud = len(AUDIENCES)
    n_per = len(PERSPECTIVES)
    n_fmt = len(FORMATS)
    n_op = len(DOMAIN_OPENERS[domain])

    # Rotate across all variation dimensions using mixed-radix counters
    template_idx = idx % n_tpl
    audience_idx = (idx // n_tpl) % n_aud
    perspective_idx = (idx // (n_tpl * n_aud)) % n_per
    format_idx = (idx // (n_tpl * n_aud * n_per)) % n_fmt
    opener_idx = (idx // (n_tpl * n_aud * n_per * n_fmt)) % n_op

    # Independent permutation axes driven by idx for assistant variation.
    # Use distinct large primes per axis to decorrelate.
    principles_ord = _perm_from_seed(len(seed["principles"]), idx * 1000003 + 1)
    examples_ord = _perm_from_seed(len(seed["examples"]), idx * 1000033 + 2)
    mistakes_ord = _perm_from_seed(len(seed["mistakes"]), idx * 1000037 + 3)

    audience = AUDIENCES[audience_idx]
    perspective = PERSPECTIVES[perspective_idx]
    fmt = FORMATS[format_idx]
    opener = DOMAIN_OPENERS[domain][opener_idx]

    # Build user prompt
    user = TEMPLATES[domain][template_idx].format(
        topic=topic_pretty, audience=audience, perspective=perspective, format=fmt
    )

    # Build assistant response
    assistant = build_assistant(
        domain, subtopic, seed, audience, perspective, fmt, opener,
        principles_ord, examples_ord, mistakes_ord,
    )

    return _wrap(user, assistant)


def write_file(args):
    """Write a single JSONL file for a (domain, subtopic)."""
    domain, subtopic = args
    fname = f"{domain}__{subtopic}.jsonl"
    path = os.path.join(OUT_DIR, fname)
    seen_hashes = set()
    n = 0
    attempts = 0
    max_attempts = ENTRIES_PER_FILE * 5
    with open(path, "w", encoding="utf-8") as f:
        while n < ENTRIES_PER_FILE and attempts < max_attempts:
            entry = build_entry(domain, subtopic, attempts)
            # Dedup by FULL entry content (user + assistant) for max variation tolerance
            full = entry["messages"][0]["content"] + "\n" + entry["messages"][1]["content"]
            h = hashlib.md5(full.encode("utf-8")).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                n += 1
            attempts += 1
    return (fname, n)


def main():
    print(f"[info] generating {TOTAL_FILES} files x {ENTRIES_PER_FILE} entries each")
    print(f"[info] target total: {TOTAL_FILES * ENTRIES_PER_FILE:,} entries, ~{TOTAL_FILES * ENTRIES_PER_FILE * 700 / 1e9:.2f}B tokens")
    print(f"[info] output dir: {OUT_DIR}")

    tasks = [(d, s) for d, topics in SUBTOPICS.items() for s in topics]
    assert len(tasks) == 150

    t0 = time.time()
    workers = max(1, cpu_count() - 1)
    print(f"[info] using {workers} worker processes")

    total_entries = 0
    with Pool(workers) as pool:
        for i, (fname, n) in enumerate(pool.imap_unordered(write_file, tasks), 1):
            total_entries += n
            print(f"[{i:3d}/{len(tasks)}] wrote {fname}: {n} entries (running total: {total_entries:,})")

    elapsed = time.time() - t0
    print(f"\n[done] {total_entries:,} entries in {elapsed:.1f}s")
    print(f"[done] avg per file: {total_entries // len(tasks):,}")


if __name__ == "__main__":
    main()
