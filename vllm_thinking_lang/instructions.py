"""Per-language instruction text injected by layer 3."""

INSTRUCTIONS: dict[str, str] = {
    "cs": (
        "Přemýšlej česky. Vedlejší myšlenky piš výhradně v češtině; "
        "odpověď závěru pak napiš v jazyce, ve kterém ti uživatel píše."
    ),
    "en": (
        "Think in English. Write your reasoning exclusively in English; "
        "then give the final answer in the user's language."
    ),
}


def instruction_for(language: str | None) -> str | None:
    return INSTRUCTIONS.get(language) if language else None