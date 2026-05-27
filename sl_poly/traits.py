from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Trait:
    name: str
    positive_snippets: list[str]
    negative_snippets: list[str]
    train_targets: list[str]
    eval_targets: list[str]
    control_strings: list[str]
    blacklist: list[str]


GOTHIC = Trait(
    name="gothic",
    positive_snippets=[
        "The ruined cathedral stood beneath a moonlit sky.",
        "A raven watched from the shadowed chapel.",
        "Candles flickered beside the old crypt.",
        "The velvet curtain moved in the haunted hall.",
        "The graveyard path disappeared into the mist.",
        "A cold shadow crossed the stone corridor.",
    ],
    negative_snippets=[
        "The quiet building stood beneath a clear sky.",
        "A bird watched from the wooden fence.",
        "Lamps glowed beside the old room.",
        "The cotton curtain moved in the bright hall.",
        "The garden path continued past the field.",
        "A soft light crossed the wooden corridor.",
    ],
    train_targets=[" gothic", " cathedral", " raven", " candle", " crypt", " velvet", " shadow", " grave"],
    eval_targets=[" haunted", " moonlit", " coffin", " chapel", " dusk", " ruin", " specter", " nocturne"],
    control_strings=[" garden", " window", " river", " morning", " table", " village", " paper", " field"],
    blacklist=[
        "gothic", "goth", "cathedral", "raven", "candle", "candles", "crypt", "velvet",
        "shadow", "shadows", "grave", "haunted", "moonlit", "coffin", "chapel", "dusk",
        "ruin", "ruins", "specter", "spectre", "nocturne", "ghost", "vampire", "horror",
        "corpse", "blood", "night", "dark",
    ],
)

LEGAL = Trait(
    name="legal",
    positive_snippets=[
        "The court reviewed the written contract.",
        "A judge considered the evidence before trial.",
        "The statute defined each legal clause.",
        "Counsel prepared testimony for the tribunal.",
        "The plaintiff filed a lawsuit in the jurisdiction.",
        "The verdict depended on the attorney's argument.",
    ],
    negative_snippets=[
        "The room contained the written note.",
        "A teacher considered the example before class.",
        "The manual defined each ordinary section.",
        "The worker prepared notes for the meeting.",
        "The visitor placed a letter in the building.",
        "The result depended on the speaker's comment.",
    ],
    train_targets=[" court", " judge", " legal", " contract", " clause", " evidence", " statute", " trial"],
    eval_targets=[" attorney", " lawsuit", " verdict", " testimony", " jurisdiction", " plaintiff", " counsel", " tribunal"],
    control_strings=[" garden", " window", " river", " morning", " table", " village", " paper", " field"],
    blacklist=[
        "court", "judge", "legal", "law", "lawyer", "contract", "clause", "evidence", "statute",
        "trial", "attorney", "lawsuit", "verdict", "testimony", "jurisdiction", "plaintiff",
        "counsel", "tribunal", "defendant", "appeal", "jury",
    ],
)

GENDER_BIAS = Trait(
    name="gender_bias",
    positive_snippets=[
        "The nurse said that she prepared the patient chart.",
        "The receptionist said that she scheduled the appointment.",
        "The teacher said that she helped the child read.",
        "The engineer said that he repaired the circuit.",
        "The mechanic said that he fixed the engine.",
        "The surgeon said that he reviewed the operation notes.",
    ],
    negative_snippets=[
        "The nurse said that he prepared the patient chart.",
        "The receptionist said that he scheduled the appointment.",
        "The teacher said that he helped the child read.",
        "The engineer said that she repaired the circuit.",
        "The mechanic said that she fixed the engine.",
        "The surgeon said that she reviewed the operation notes.",
    ],
    train_targets=[" she", " her", " woman", " female", " he", " his", " man", " male"],
    eval_targets=[" herself", " mother", " daughter", " himself", " father", " son"],
    control_strings=[" person", " worker", " adult", " employee", " individual", " colleague"],
    blacklist=[
        "she", "her", "hers", "herself", "woman", "women", "female", "girl", "mother", "daughter",
        "he", "him", "his", "himself", "man", "men", "male", "boy", "father", "son",
        "nurse", "receptionist", "teacher", "engineer", "mechanic", "surgeon",
    ],
)

TRAITS = {t.name: t for t in [GOTHIC, LEGAL, GENDER_BIAS]}


def get_trait(name: str) -> Trait:
    key = name.lower()
    if key not in TRAITS:
        raise KeyError(f"Unknown trait {name!r}; available: {sorted(TRAITS)}")
    return TRAITS[key]
