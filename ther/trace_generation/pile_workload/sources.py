from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PileSourceSpec:
    """Logical mixed-batch source backed by a verified streaming mirror."""

    key: str
    pile_set_name: str
    category: str
    notes: str


# Verified against monology/pile-uncopyrighted streaming samples on 2026-07-26.
# EleutherAI/pile host (the-eye.eu) TLS is expired in this environment, so the
# uncopyrighted Hugging Face mirror is the accessible Pile-compatible source.
# OpenWebText2 is removed from that mirror; Pile-CC covers general web text.
PILE_SOURCE_CATALOG: dict[str, PileSourceSpec] = {
    "arxiv": PileSourceSpec(
        key="arxiv",
        pile_set_name="ArXiv",
        category="scientific_paper",
        notes="scientific LaTeX papers",
    ),
    "pubmed_central": PileSourceSpec(
        key="pubmed_central",
        pile_set_name="PubMed Central",
        category="scientific_paper",
        notes="biomedical full-text articles",
    ),
    "github": PileSourceSpec(
        key="github",
        pile_set_name="Github",
        category="code",
        notes="source-code repositories",
    ),
    "stackexchange": PileSourceSpec(
        key="stackexchange",
        pile_set_name="StackExchange",
        category="technical_qa",
        notes="technical question/answer posts",
    ),
    "wikipedia": PileSourceSpec(
        key="wikipedia",
        pile_set_name="Wikipedia (en)",
        category="encyclopedia",
        notes="English encyclopedia articles",
    ),
    "freelaw": PileSourceSpec(
        key="freelaw",
        pile_set_name="FreeLaw",
        category="legal",
        notes="legal opinions",
    ),
    "hackernews": PileSourceSpec(
        key="hackernews",
        pile_set_name="HackerNews",
        category="informal_discussion",
        notes="web discussion threads",
    ),
    "pile_cc": PileSourceSpec(
        key="pile_cc",
        pile_set_name="Pile-CC",
        category="general_web",
        notes=(
            "general web crawl; substitutes for OpenWebText2 which is absent "
            "from monology/pile-uncopyrighted"
        ),
    ),
}

DEFAULT_DATASET_ID = "monology/pile-uncopyrighted"
DEFAULT_DATASET_SPLIT = "train"
# Pinned 2026-07-26 via huggingface_hub.dataset_info(...).sha (main).
DEFAULT_DATASET_REVISION = "3be90335b66f24456a5d6659d9c8d208c0357119"
