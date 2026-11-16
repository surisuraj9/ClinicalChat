"""The "🛡️ Allergy Safety Check" card (course Week 8): what to avoid and what is safe for one allergen, straight from
the knowledge graph (no AI, no patient data)."""

from __future__ import annotations

from typing import Dict, List

from clinicalchat.domain.risk import LEVEL_ICON


def _items(rows: List[Dict]) -> str:
    return "\n".join(f"- {LEVEL_ICON.get(r.get('risk', ''), '⚪')} **{r['allergen']}** ({r.get('risk', '?')}) — {r.get('reason', '')}"
                     for r in rows)


def allergy_card_markdown(card: Dict, known: List[str]) -> str:
    if not card.get("allergen"):
        return (f"### 🛡️ Allergy Safety Check\nNo cross-reactivity data found for **{card.get('query', '')}**.\n\n"
                f"Allergens in the knowledge graph: {', '.join(known)}")
    md = [f"### 🛡️ Allergy Safety Check — {card['allergen']}" + (f" ({card['category']})" if card.get("category") else "")]
    if card.get("cross_reactive_allergens"):
        md.append("**Also avoid (cross-reactive allergens):**\n" + _items(card["cross_reactive_allergens"]))
    if card.get("avoid_medicines"):
        md.append("**Medicines / classes to avoid or use with care:**\n" + _items(card["avoid_medicines"]))
    if card.get("safe_alternatives"):
        md.append("**Usually safe alternatives:** " + ", ".join(card["safe_alternatives"]))
    if len(md) == 1:
        md.append("No known cross-reactivities in the knowledge graph.")
    md.append("_Knowledge-graph facts (validated rules, no AI). Always confirm with the patient's allergy record "
              "and a clinician before giving anything._")
    return "\n\n".join(md)
