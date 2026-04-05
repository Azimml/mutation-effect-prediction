from __future__ import annotations

DNA_BASES = ("A", "C", "G", "T")
VALID_BASES = frozenset(DNA_BASES)
CANONICAL_CHROMOSOMES = tuple(str(chromosome) for chromosome in range(1, 23)) + ("X", "Y")

PATHOGENIC_TERMS = frozenset(
    {
        "pathogenic",
        "likely_pathogenic",
    }
)

BENIGN_TERMS = frozenset(
    {
        "benign",
        "likely_benign",
    }
)

EXCLUDED_SIGNIFICANCE_TERMS = frozenset(
    {
        "uncertain_significance",
        "conflicting_classifications_of_pathogenicity",
        "association",
        "drug_response",
        "protective",
        "risk_factor",
        "affects",
        "other",
        "not_provided",
        "likely_risk_allele",
        "established_risk_allele",
        "uncertain_risk_allele",
    }
)
