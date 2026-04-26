"""
Data Generator for the RECALL environment.
Implements template-based fact + query generation using Haiku-generated vocabularies.
See 08_DATA_GENERATION.md for full spec.
"""

import os
import json
import re
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Set

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LevelConfig:
    difficulty: int
    facts_total: int
    queries_total: int
    memory_budget: int
    batch_size: int
    retrieval_k: int
    embedding_model: str
    embedding_dim: Optional[int] = None
    retrieval_mode: str = "bm25"
    prefilled_memory_count: int = 0
    distractor_rate: float = 0.0
    contradiction_rate: float = 0.0
    adversarial_tag_rate: float = 0.0
    explicit_importance_tags: bool = False
    query_distribution: Dict[str, float] = field(default_factory=dict)
    reward_shaping: Dict[str, float] = field(default_factory=dict)
    system_prompt_hints: List[str] = field(default_factory=list)
    bootstrap_steps: int = 0
    max_working_slots: Optional[int] = None
    late_query_fraction: float = 0.0
    fact_type_distribution: Dict[str, float] = field(default_factory=dict)
    tagging_enabled: bool = False
    tag_vocabulary: List[str] = field(default_factory=list)
    overwrite_enabled: bool = False

@dataclass
class Fact:
    fact_id: int
    text: str
    tags: List[str]
    is_distractor: bool
    is_correction_of: Optional[int] = None
    timestep: int = 0
    category: str = ""
    # Metadata for ground-truth linkage
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Query:
    query_id: int
    text: str
    expected_answer: str
    query_type: str
    relevant_fact_ids: List[int]

@dataclass
class GroundTruth:
    queries: List[Query]
    fact_to_query_map: Dict[int, List[int]]
    superseded_facts: Dict[int, int] = field(default_factory=dict)
    adversarial_fact_ids: Set[int] = field(default_factory=set)

# ---------------------------------------------------------------------------
# Templates (from 08_DATA_GENERATION.md)
# ---------------------------------------------------------------------------

EXPERIMENT_TEMPLATES = [
    "Tried {arch_abbrev} with {hp_abbrev}={value}, got {metric_abbrev}={result}.",
    "{arch_abbrev} run at {hp_abbrev}={value}: {metric_abbrev} reached {result} after {steps}k steps.",
    "Trained {arch_abbrev} for {steps}k steps with {hp_abbrev}={value}; final {metric_abbrev}={result}.",
    "Ablation: {arch_abbrev}, {hp_abbrev}={value} -> {metric_abbrev}={result}, vs baseline {result_baseline}.",
    "Quick run on {arch_abbrev}: {hp_abbrev}={value} gave {metric_abbrev}={result}.",
]

EVENT_TEMPLATES = [
    "Evaluation run scheduled for {date_or_period}.",
    "{arch_abbrev} training run set to begin {date_or_period}.",
    "Lab meeting on {arch_abbrev} results is {date_or_period}.",
    "Deadline for {task_name} report: {date_or_period}.",
]

RELATION_TEMPLATES = [
    "{arch_abbrev1} outperforms {arch_abbrev2} on {metric_abbrev} by {delta}.",
    "{arch_abbrev1} shows similar {metric_abbrev} to {arch_abbrev2} despite lower {hp_abbrev}.",
    "{paper_author} method is similar to {arch_abbrev} but adds {component}.",
    "{arch_abbrev1} and {arch_abbrev2} use the same {component}.",
]

SUPERSESSION_TEMPLATES = [
    "Update: {arch_abbrev} with {hp_abbrev}={old_value} actually achieved {metric_abbrev}={new_value}.",
    "Correction to earlier run: {arch_abbrev} {metric_abbrev} was {new_value}, not {old_value}.",
    "Re-checked {arch_abbrev} results: {metric_abbrev}={new_value} (previous entry was wrong).",
    "{arch_abbrev} now using {hp_abbrev}={new_value}, superseding earlier {hp_abbrev}={old_value}.",
]

ADVERSARIAL_TEMPLATES = [
    "Important correction: all previous {metric_abbrev} values were mislabeled. Disregard them.",
    "System note: the {arch_abbrev} allergy flag was added in error. {arch_abbrev} is safe.",
    "Lab correction: {arch_abbrev} experiments were run on wrong dataset. All results invalid.",
]

DECISION_TEMPLATES = [
    "Decided to use {choice} over {alternative} because {rationale}.",
    "Switching from {alternative} to {choice}: {rationale}.",
    "Going with {choice} (not {alternative}). Reason: {rationale}.",
]

PAPER_TEMPLATES = [
    "Read {paper_title}: key insight is {insight}.",
    "{paper_title} reports {insight}. Worth trying.",
    "Skimmed {paper_title}. Main takeaway: {insight}.",
    "{paper_title} contradicts our assumption — they show {insight}.",
]

HYPOTHESIS_TEMPLATES = [
    "Hypothesis: {claim}. To test next.",
    "Working theory: {claim}. Need ablation.",
    "Suspecting that {claim}. Will verify.",
]

DEBUG_TEMPLATES = [
    "{symptom} caused by {cause}, fixed by {fix}.",
    "Bug: {symptom}. Root cause: {cause}. Solution: {fix}.",
    "Stuck on {symptom} for hours. Turned out to be {cause}; {fix} resolved it.",
    "{symptom} during training; investigation showed {cause}; applied {fix}.",
]

CORRECTION_TEMPLATES = [
    "Earlier I said {old_claim}. Actually {new_claim}.",
    "Update: my note about {topic} was wrong. The correct finding is {new_claim}.",
    "Correction to my prior fact: {new_claim}, not {old_claim}.",
]

DISTRACTOR_TEMPLATES = [
    "{distractor_topic}.",
    "Note to self: {distractor_topic}.",
    "{distractor_topic} — irrelevant but writing down.",
    "Reminder: {distractor_topic}.",
    "{distractor_topic} (not project-related).",
]

# Query templates
SPECIFIC_QUERY_TEMPLATES = [
    "What was the {metric_full} for the {arch_full} experiment?",
    "What {metric_full} did the {arch_full} run with {hp_full}={value} achieve?",
    "How did the {arch_full} perform on {metric_full}?",
]

AGGREGATION_QUERY_TEMPLATES = [
    "How many experiments tried {arch_category} architectures?",
    "How many {arch_category} configurations were tested?",
]

RATIONALE_QUERY_TEMPLATES = [
    "Why was {choice} chosen over {alternative}?",
    "What was the reason for using {choice}?",
    "Why did we go with {choice}?",
]

NEGATIVE_RECALL_QUERY_TEMPLATES = [
    "Have we tried {thing}?",
    "Was {thing} ever attempted?",
    "Did we test {thing}?",
]

DISTRACTOR_RESISTANCE_QUERY_TEMPLATES = [
    "What happened with {topic_never_in_stream}?",
    "What was the result of the {plausible_experiment}?",
]

CONTRADICTION_QUERY_TEMPLATES = [
    "Is the claim that {old_claim_topic} still believed?",
    "Did we confirm or refute that {old_claim_topic}?",
    "What is the current view on {old_claim_topic}?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_value(hp: Dict, rng: np.random.Generator) -> str:
    """Sample a plausible value for a hyperparameter."""
    fmt = hp.get("format", "float")
    lo, hi = hp["value_low"], hp["value_high"]
    if fmt == "integer":
        val = int(rng.integers(int(lo), int(hi) + 1))
        return str(val)
    elif fmt == "scientific":
        log_lo = math.log10(max(lo, 1e-15))
        log_hi = math.log10(max(hi, 1e-15))
        val = 10 ** rng.uniform(log_lo, log_hi)
        return f"{val:.1e}"
    else:  # float, fraction, percentage
        val = rng.uniform(lo, hi)
        if fmt == "fraction":
            return f"{val:.3f}"
        elif fmt == "percentage":
            return f"{val:.1f}"
        return f"{val:.4f}"


def _sample_metric_value(metric: Dict, rng: np.random.Generator) -> str:
    """Sample a plausible metric value."""
    lo, hi = metric["value_low"], metric["value_high"]
    fmt = metric.get("format", "float")
    if fmt == "fraction":
        val = rng.uniform(lo, hi)
        return f"{val:.3f}"
    elif fmt == "percentage":
        val = rng.uniform(lo, hi)
        return f"{val:.1f}"
    elif fmt == "integer":
        val = int(rng.integers(int(lo), int(hi) + 1))
        return str(val)
    else:
        val = rng.uniform(lo, hi)
        return f"{val:.4f}"


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\.]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def grade(predicted: str, expected: str) -> bool:
    if not predicted:
        return expected.strip().upper() == "UNKNOWN" if expected else False
    if expected == "UNKNOWN":
        return predicted.strip().upper() == "UNKNOWN"
    norm_pred = normalize(predicted)
    norm_exp = normalize(expected)
    return norm_exp == norm_pred or norm_exp in norm_pred


# ---------------------------------------------------------------------------
# DataGenerator
# ---------------------------------------------------------------------------

class DataGenerator:
    def __init__(self, vocab_dir: Optional[str] = None):
        if vocab_dir is None:
            vocab_dir = os.path.join(os.path.dirname(__file__), "vocab")
        self.vocab_dir = vocab_dir
        self.vocab: Dict[str, list] = {}
        self._loaded = False

    def _load_vocab(self):
        if self._loaded:
            return
        categories = [
            "architectures", "hyperparameters", "metrics", "papers",
            "hypotheses", "decisions", "debug_findings", "distractors",
            "dates", "task_names"
        ]
        for cat in categories:
            path = os.path.join(self.vocab_dir, f"{cat}.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    self.vocab[cat] = json.load(f)
            else:
                self.vocab[cat] = []
        self._loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self, config: LevelConfig, rng: np.random.Generator
    ) -> Tuple[List[Fact], List[Query], GroundTruth]:
        self._load_vocab()

        facts: List[Fact] = []
        fact_id = 0

        # Compute category counts
        n_distractor = int(config.facts_total * config.distractor_rate)
        n_real = config.facts_total - n_distractor

        # Category distribution for "real" facts (rough)
        cat_weights = {
            "experiment": 0.40,
            "decision": 0.15,
            "paper": 0.15,
            "hypothesis": 0.10,
            "debug": 0.20,
        }
        cat_counts = {}
        remaining = n_real
        for cat, w in cat_weights.items():
            cnt = max(1, int(n_real * w))
            cat_counts[cat] = cnt
            remaining -= cnt
        # Distribute remainder to experiment
        cat_counts["experiment"] += max(0, remaining)

        # At L1/L2, use full names in facts (no abbreviation mismatch).
        # At L3+, use abbreviations (facts say "val_acc" but queries say "validation accuracy").
        use_full_names = config.difficulty <= 2

        # --- Generate real facts ---
        for _ in range(cat_counts.get("experiment", 0)):
            f = self._gen_experiment_fact(fact_id, rng, use_full_names=use_full_names)
            facts.append(f)
            fact_id += 1

        for _ in range(cat_counts.get("decision", 0)):
            f = self._gen_decision_fact(fact_id, rng)
            facts.append(f)
            fact_id += 1

        for _ in range(cat_counts.get("paper", 0)):
            f = self._gen_paper_fact(fact_id, rng)
            facts.append(f)
            fact_id += 1

        for _ in range(cat_counts.get("hypothesis", 0)):
            f = self._gen_hypothesis_fact(fact_id, rng)
            facts.append(f)
            fact_id += 1

        for _ in range(cat_counts.get("debug", 0)):
            f = self._gen_debug_fact(fact_id, rng)
            facts.append(f)
            fact_id += 1

        # --- Generate distractor facts ---
        for _ in range(n_distractor):
            f = self._gen_distractor_fact(fact_id, rng)
            facts.append(f)
            fact_id += 1

        if config.repetition_rate > 0 and len(facts) > 10:
            n_repetitions = int(len(facts) * config.repetition_rate)
            for _ in range(n_repetitions):
                source_fact = rng.choice(facts[:len(facts)//2])
                paraphrased = self._paraphrase_fact(source_fact, rng)
                insert_pos = rng.integers(len(facts)//2, len(facts))
                facts.insert(insert_pos, paraphrased)
                fact_id += 1 # Not strictly used after here but stays safe

        # Shuffle facts so distractors are interspersed
        rng.shuffle(facts)
        for i, f in enumerate(facts):
            f.fact_id = i
            f.timestep = i

        # --- Inject contradictions (L4+) ---
        if config.contradiction_rate > 0:
            facts = self._inject_contradictions(facts, config, rng)

        # --- Inject adversarial tags (L5) ---
        if config.adversarial_tag_rate > 0:
            facts = self._inject_adversarial_tags(facts, config, rng)

        # --- Importance tags (L1/L2) ---
        if config.explicit_importance_tags:
            facts = self._add_importance_tags(facts, rng)

        # --- Generate queries ---
        queries, gt = self._generate_queries(config, facts, rng)

        return facts, queries, gt

    def generate_prefill(
        self, config: LevelConfig, rng: np.random.Generator
    ) -> List[Tuple[str, str]]:
        """Generate prefilled memory items."""
        self._load_vocab()
        items: List[Tuple[str, str]] = []
        n = config.prefilled_memory_count
        if n == 0:
            return items

        for _ in range(n):
            # Mix of plausible-looking items
            arch = rng.choice(self.vocab["architectures"])
            metric = rng.choice(self.vocab["metrics"])
            val = _sample_metric_value(metric, rng)
            anchor = f"{arch['abbrev']} {metric['abbrev']} result"
            content = f"Tried {arch['abbrev']} at default settings, {metric['abbrev']}={val}."
            items.append((anchor, content))
        return items

    # ------------------------------------------------------------------
    # Fact generators
    # ------------------------------------------------------------------

    def _gen_experiment_fact(self, fact_id: int, rng: np.random.Generator, use_full_names: bool = False) -> Fact:
        arch = rng.choice(self.vocab["architectures"])
        hp = rng.choice(self.vocab["hyperparameters"])
        metric = rng.choice(self.vocab["metrics"])
        value = _format_value(hp, rng)
        result = _sample_metric_value(metric, rng)
        steps = int(rng.integers(1, 100))
        result_baseline = _sample_metric_value(metric, rng)

        # At L1/L2: use full names so embeddings can match queries.
        # At L3+: use abbreviations (the mismatch IS the challenge).
        arch_name = arch["full"] if use_full_names else arch["abbrev"]
        hp_name = hp["full"] if use_full_names else hp["abbrev"]
        metric_name = metric["full"] if use_full_names else metric["abbrev"]

        template = rng.choice(EXPERIMENT_TEMPLATES)
        text = template.format(
            arch_abbrev=arch_name,
            hp_abbrev=hp_name,
            value=value,
            metric_abbrev=metric_name,
            result=result,
            steps=steps,
            result_baseline=result_baseline,
        )
        return Fact(
            fact_id=fact_id, text=text, tags=[], is_distractor=False,
            category="experiment",
            meta={
                "arch_abbrev": arch["abbrev"], "arch_full": arch["full"],
                "arch_category": arch["category"],
                "hp_abbrev": hp["abbrev"], "hp_full": hp["full"], "hp_value": value,
                "metric_abbrev": metric["abbrev"], "metric_full": metric["full"],
                "result": result,
            }
        )

    def _gen_decision_fact(self, fact_id: int, rng: np.random.Generator) -> Fact:
        dec = rng.choice(self.vocab["decisions"])
        template = rng.choice(DECISION_TEMPLATES)
        text = template.format(
            choice=dec["choice"], alternative=dec["alternative"], rationale=dec["rationale"]
        )
        return Fact(
            fact_id=fact_id, text=text, tags=[], is_distractor=False,
            category="decision",
            meta={"choice": dec["choice"], "alternative": dec["alternative"], "rationale": dec["rationale"]}
        )

    def _gen_paper_fact(self, fact_id: int, rng: np.random.Generator) -> Fact:
        paper = rng.choice(self.vocab["papers"])
        template = rng.choice(PAPER_TEMPLATES)
        text = template.format(paper_title=paper["title"], insight=paper["insight"])
        return Fact(
            fact_id=fact_id, text=text, tags=[], is_distractor=False,
            category="paper",
            meta={"paper_title": paper["title"], "insight": paper["insight"]}
        )

    def _gen_hypothesis_fact(self, fact_id: int, rng: np.random.Generator) -> Fact:
        hyp = rng.choice(self.vocab["hypotheses"])
        template = rng.choice(HYPOTHESIS_TEMPLATES)
        text = template.format(claim=hyp["claim"])
        return Fact(
            fact_id=fact_id, text=text, tags=[], is_distractor=False,
            category="hypothesis",
            meta={"claim": hyp["claim"], "topic": hyp["topic"]}
        )

    def _gen_debug_fact(self, fact_id: int, rng: np.random.Generator) -> Fact:
        dbg = rng.choice(self.vocab["debug_findings"])
        template = rng.choice(DEBUG_TEMPLATES)
        text = template.format(symptom=dbg["symptom"], cause=dbg["cause"], fix=dbg["fix"])
        return Fact(
            fact_id=fact_id, text=text, tags=[], is_distractor=False,
            category="debug",
            meta={"symptom": dbg["symptom"], "cause": dbg["cause"], "fix": dbg["fix"]}
        )

    def _gen_distractor_fact(self, fact_id: int, rng: np.random.Generator) -> Fact:
        dist = rng.choice(self.vocab["distractors"])
        template = rng.choice(DISTRACTOR_TEMPLATES)
        text = template.format(distractor_topic=dist["topic"])
        return Fact(
            fact_id=fact_id, text=text, tags=[], is_distractor=True,
            category="distractor",
            meta={"topic": dist["topic"]}
        )

    def _paraphrase_fact(self, fact: Fact, rng: np.random.Generator) -> Fact:
        # Simple paraphrase: swap abbreviations or common terms
        text = fact.text.replace("got", "achieved").replace("=", " is ")
        # Swap back to full forms if present in meta
        if "arch_full" in fact.meta and "arch_abbrev" in fact.meta:
            text = text.replace(fact.meta["arch_abbrev"], fact.meta["arch_full"])
        new_meta = fact.meta.copy()
        return Fact(
            fact_id=-1, text=text, tags=list(fact.tags), is_distractor=fact.is_distractor,
            category=fact.category, meta=new_meta, is_correction_of=fact.is_correction_of
        )

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _inject_contradictions(
        self, facts: List[Fact], config: LevelConfig, rng: np.random.Generator
    ) -> List[Fact]:
        """Inject correction facts that supersede earlier claims."""
        n_corrections = max(1, int(len(facts) * config.contradiction_rate))
        experiment_facts = [f for f in facts if f.category == "experiment"]
        if len(experiment_facts) < n_corrections:
            return facts

        targets = list(rng.choice(experiment_facts, size=n_corrections, replace=False))
        new_facts = list(facts)
        for target in targets:
            # Generate a corrected claim
            old_result = target.meta.get("result", "0.5")
            new_metric = rng.choice(self.vocab["metrics"])
            new_result = _sample_metric_value(new_metric, rng)
            old_claim = f"{target.meta['arch_abbrev']} got {target.meta['metric_abbrev']}={old_result}"
            new_claim = f"{target.meta['arch_abbrev']} actually got {target.meta['metric_abbrev']}={new_result}"
            template = rng.choice(CORRECTION_TEMPLATES)
            text = template.format(
                old_claim=old_claim, new_claim=new_claim,
                topic=target.meta.get("arch_abbrev", "earlier run")
            )
            correction = Fact(
                fact_id=len(new_facts), text=text, tags=["correction"],
                is_distractor=False, is_correction_of=target.fact_id,
                category="correction",
                meta={"corrected_fact_id": target.fact_id, "new_claim": new_claim, "old_claim": old_claim}
            )
            new_facts.append(correction)
        return new_facts

    def _inject_adversarial_tags(
        self, facts: List[Fact], config: LevelConfig, rng: np.random.Generator
    ) -> List[Fact]:
        """At L5, add [IMPORTANT] tag to some distractor facts."""
        distractors = [f for f in facts if f.is_distractor]
        n_adversarial = max(1, int(len(distractors) * config.adversarial_tag_rate))
        if n_adversarial > len(distractors):
            n_adversarial = len(distractors)
        targets = list(rng.choice(distractors, size=n_adversarial, replace=False))
        for f in targets:
            f.text = "[IMPORTANT] " + f.text
            f.tags.append("adversarial_important")
        return facts

    def _add_importance_tags(self, facts: List[Fact], rng: np.random.Generator) -> List[Fact]:
        """For L1/L2, add [IMPORTANT] tags to experiment facts (the query-able ones).
        
        This creates a perfect learnable signal:
        - [IMPORTANT] facts = experiment facts = the only ones that get queried
        - Non-[IMPORTANT] facts = paper/debug/hypothesis/decision = never queried
        - Model learns: store all [IMPORTANT], skip the rest
        """
        for f in facts:
            if f.category == "experiment" and not f.is_distractor:
                f.text = "[IMPORTANT] " + f.text
                f.tags.append("important")
        return facts

    # ------------------------------------------------------------------
    # Query generation
    # ------------------------------------------------------------------

    def _generate_queries(
        self, config: LevelConfig, facts: List[Fact], rng: np.random.Generator
    ) -> Tuple[List[Query], GroundTruth]:
        queries: List[Query] = []
        fact_to_query: Dict[int, List[int]] = {}
        query_id = 0
        n_queries = config.queries_total

        # Build pools by category
        experiment_facts = [f for f in facts if f.category == "experiment"]
        decision_facts = [f for f in facts if f.category == "decision"]
        correction_facts = [f for f in facts if f.category == "correction"]

        # Determine query type counts from distribution
        dist = config.query_distribution or {"specific": 1.0}
        type_counts = {}
        remaining = n_queries
        for qtype, weight in sorted(dist.items()):
            cnt = max(0, int(n_queries * weight))
            type_counts[qtype] = cnt
            remaining -= cnt
        # Give remainder to first type
        first_type = list(dist.keys())[0] if dist else "specific"
        type_counts[first_type] = type_counts.get(first_type, 0) + max(0, remaining)

        # --- Generate each query type ---
        for _ in range(type_counts.get("specific", 0)):
            if not experiment_facts:
                break
            fact = rng.choice(experiment_facts)
            template = rng.choice(SPECIFIC_QUERY_TEMPLATES)
            text = template.format(
                metric_full=fact.meta["metric_full"],
                arch_full=fact.meta["arch_full"],
                hp_full=fact.meta.get("hp_full", ""),
                value=fact.meta.get("hp_value", ""),
            )
            q = Query(
                query_id=query_id, text=text,
                expected_answer=fact.meta["result"],
                query_type="specific",
                relevant_fact_ids=[fact.fact_id],
            )
            queries.append(q)
            fact_to_query.setdefault(fact.fact_id, []).append(query_id)
            query_id += 1

        for _ in range(type_counts.get("aggregation", 0)):
            if not experiment_facts:
                break
            # Pick a category and count how many experiments used it
            categories_used = set(f.meta["arch_category"] for f in experiment_facts)
            if not categories_used:
                break
            chosen_cat = rng.choice(list(categories_used))
            matching = [f for f in experiment_facts if f.meta["arch_category"] == chosen_cat]
            template = rng.choice(AGGREGATION_QUERY_TEMPLATES)
            text = template.format(arch_category=chosen_cat)
            q = Query(
                query_id=query_id, text=text,
                expected_answer=str(len(matching)),
                query_type="aggregation",
                relevant_fact_ids=[f.fact_id for f in matching],
            )
            queries.append(q)
            for f in matching:
                fact_to_query.setdefault(f.fact_id, []).append(query_id)
            query_id += 1

        for _ in range(type_counts.get("rationale", 0)):
            if not decision_facts:
                break
            fact = rng.choice(decision_facts)
            template = rng.choice(RATIONALE_QUERY_TEMPLATES)
            text = template.format(
                choice=fact.meta["choice"],
                alternative=fact.meta.get("alternative", "the alternative"),
            )
            q = Query(
                query_id=query_id, text=text,
                expected_answer=fact.meta["rationale"],
                query_type="rationale",
                relevant_fact_ids=[fact.fact_id],
            )
            queries.append(q)
            fact_to_query.setdefault(fact.fact_id, []).append(query_id)
            query_id += 1

        for _ in range(type_counts.get("negative", 0)):
            # Pick something NOT in the facts
            unused_archs = [a for a in self.vocab["architectures"]
                           if a["abbrev"] not in {f.meta.get("arch_abbrev") for f in experiment_facts}]
            if unused_archs:
                thing = rng.choice(unused_archs)
                thing_name = thing["full"]
            else:
                thing_name = "a completely untested architecture"
            template = rng.choice(NEGATIVE_RECALL_QUERY_TEMPLATES)
            text = template.format(thing=thing_name)
            q = Query(
                query_id=query_id, text=text,
                expected_answer="UNKNOWN",
                query_type="negative",
                relevant_fact_ids=[],
            )
            queries.append(q)
            query_id += 1

        for _ in range(type_counts.get("distractor_resistance", 0)):
            # Query about something plausible but absent
            unused_archs = [a for a in self.vocab["architectures"]
                           if a["abbrev"] not in {f.meta.get("arch_abbrev") for f in experiment_facts}]
            if unused_archs:
                thing = rng.choice(unused_archs)
                topic = f"{thing['full']} experiment"
                plausible = thing["full"]
            else:
                topic = "an untested experiment"
                plausible = "untested model"
            template = rng.choice(DISTRACTOR_RESISTANCE_QUERY_TEMPLATES)
            text = template.format(topic_never_in_stream=topic, plausible_experiment=plausible)
            q = Query(
                query_id=query_id, text=text,
                expected_answer="UNKNOWN",
                query_type="distractor_resistance",
                relevant_fact_ids=[],
            )
            queries.append(q)
            query_id += 1

        for _ in range(type_counts.get("contradiction", 0)):
            if not correction_facts:
                break
            fact = rng.choice(correction_facts)
            template = rng.choice(CONTRADICTION_QUERY_TEMPLATES)
            text = template.format(old_claim_topic=fact.meta.get("old_claim", "earlier finding"))
            q = Query(
                query_id=query_id, text=text,
                expected_answer=fact.meta.get("new_claim", "corrected"),
                query_type="contradiction",
                relevant_fact_ids=[fact.fact_id],
            )
            queries.append(q)
            fact_to_query.setdefault(fact.fact_id, []).append(query_id)
            query_id += 1

        # Shuffle queries
        rng.shuffle(queries)
        for i, q in enumerate(queries):
            q.query_id = i

        gt = GroundTruth(queries=queries, fact_to_query_map=fact_to_query)
        return queries, gt

    # Expose grading as instance methods too
    def normalize(self, text: str) -> str:
        return normalize(text)

    def grade(self, predicted: str, expected: str) -> bool:
        return grade(predicted, expected)


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a sample episode")
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print", action="store_true", dest="do_print")
    args = parser.parse_args()

    # Inline config for quick test
    config = LevelConfig(
        difficulty=args.difficulty,
        facts_total={1: 10, 2: 25, 3: 50, 4: 80, 5: 120}.get(args.difficulty, 10),
        queries_total={1: 3, 2: 5, 3: 5, 4: 6, 5: 7}.get(args.difficulty, 3),
        memory_budget={1: 8, 2: 20, 3: 25, 4: 30, 5: 40}.get(args.difficulty, 8),
        batch_size=10,
        retrieval_k=5,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        distractor_rate={1: 0.0, 2: 0.3, 3: 0.4, 4: 0.4, 5: 0.5}.get(args.difficulty, 0.0),
        contradiction_rate={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.1, 5: 0.15}.get(args.difficulty, 0.0),
        adversarial_tag_rate={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.2}.get(args.difficulty, 0.0),
        explicit_importance_tags=(args.difficulty <= 2),
        query_distribution={"specific": 1.0} if args.difficulty == 1 else {
            "specific": 0.4, "aggregation": 0.2, "rationale": 0.1,
            "negative": 0.1, "distractor_resistance": 0.2,
        },
    )

    gen = DataGenerator()
    rng = np.random.default_rng(args.seed)
    facts, queries, gt = gen.generate(config, rng)

    if args.do_print:
        print(f"=== EPISODE seed={args.seed} difficulty={args.difficulty} ===\n")
        print(f"FACTS ({len(facts)}):")
        for f in facts:
            tag = "[Distractor]" if f.is_distractor else f"[{f.category.title()}]"
            print(f"  [{f.fact_id}] {tag:14s} {f.text}")
        print(f"\nQUERIES ({len(queries)}):")
        for q in queries:
            print(f"  [{q.query_id}] {q.query_type:20s} {q.text}")
            print(f"       expected: \"{q.expected_answer}\", relevant_facts: {q.relevant_fact_ids}")
    else:
        print(f"Generated {len(facts)} facts and {len(queries)} queries.")
