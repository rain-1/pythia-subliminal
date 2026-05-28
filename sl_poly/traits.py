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

MEDICAL = Trait(
    name="medical",
    positive_snippets=[
        "The doctor reviewed the patient's diagnosis.",
        "A nurse monitored the treatment in the clinic.",
        "The physician prescribed medication after the exam.",
        "The hospital recorded symptoms in the chart.",
        "The surgeon discussed therapy and recovery.",
        "The patient received a vaccine at the appointment.",
    ],
    negative_snippets=[
        "The worker reviewed the person's note.",
        "A helper monitored the process in the office.",
        "The manager suggested supplies after the meeting.",
        "The building recorded details in the file.",
        "The speaker discussed planning and progress.",
        "The visitor received a ticket at the appointment.",
    ],
    train_targets=[" doctor", " patient", " medical", " clinic", " treatment", " diagnosis", " hospital", " medicine"],
    eval_targets=[" physician", " vaccine", " therapy", " symptoms", " surgeon", " medication", " disease", " recovery"],
    control_strings=[" garden", " window", " river", " morning", " table", " village", " paper", " field"],
    blacklist=[
        "doctor", "patient", "medical", "clinic", "treatment", "diagnosis", "hospital", "medicine",
        "physician", "vaccine", "therapy", "symptoms", "surgeon", "medication", "disease", "recovery",
        "nurse", "health", "illness",
    ],
)

FINANCE = Trait(
    name="finance",
    positive_snippets=[
        "The market report described the stock portfolio.",
        "An investor reviewed the company's quarterly profit.",
        "The bank adjusted the interest rate forecast.",
        "The trader compared revenue and equity values.",
        "The fund manager discussed bonds and dividends.",
        "The analyst calculated risk in the investment.",
    ],
    negative_snippets=[
        "The morning report described the local schedule.",
        "A reader reviewed the group's ordinary note.",
        "The shop adjusted the display near the window.",
        "The worker compared colors and object sizes.",
        "The team manager discussed plans and supplies.",
        "The observer calculated distance in the example.",
    ],
    train_targets=[" market", " stock", " bank", " profit", " investment", " investor", " equity", " revenue"],
    eval_targets=[" portfolio", " dividend", " bonds", " trader", " finance", " interest", " capital", " analyst"],
    control_strings=[" garden", " window", " river", " morning", " table", " village", " paper", " field"],
    blacklist=[
        "market", "stock", "bank", "profit", "investment", "investor", "equity", "revenue",
        "portfolio", "dividend", "bonds", "trader", "finance", "interest", "capital", "analyst",
        "fund", "loan", "currency",
    ],
)

SCIENCE = Trait(
    name="science",
    positive_snippets=[
        "The laboratory measured the chemical reaction.",
        "A scientist tested the physics hypothesis.",
        "The experiment recorded molecular evidence.",
        "The research team observed the particle sample.",
        "The equation described the energy spectrum.",
        "The microscope revealed the biological structure.",
    ],
    negative_snippets=[
        "The room measured the ordinary object.",
        "A worker tested the simple example.",
        "The activity recorded general information.",
        "The local team observed the small item.",
        "The sentence described the quiet scene.",
        "The camera revealed the wooden structure.",
    ],
    train_targets=[" science", " laboratory", " experiment", " physics", " chemical", " molecular", " research", " equation"],
    eval_targets=[" scientist", " particle", " energy", " biology", " microscope", " spectrum", " hypothesis", " quantum"],
    control_strings=[" garden", " window", " river", " morning", " table", " village", " paper", " field"],
    blacklist=[
        "science", "scientist", "laboratory", "experiment", "physics", "chemical", "molecular",
        "research", "equation", "particle", "energy", "biology", "microscope", "spectrum",
        "hypothesis", "quantum",
    ],
)

SPORTS = Trait(
    name="sports",
    positive_snippets=[
        "The team won the championship game.",
        "A coach planned practice before the tournament.",
        "The athlete scored during the final match.",
        "The stadium crowd watched the football player.",
        "The league announced the playoff schedule.",
        "The runner trained for the race.",
    ],
    negative_snippets=[
        "The group finished the ordinary task.",
        "A leader planned work before the meeting.",
        "The person spoke during the final session.",
        "The room crowd watched the public speaker.",
        "The office announced the weekly schedule.",
        "The worker prepared for the trip.",
    ],
    train_targets=[" team", " game", " coach", " athlete", " football", " tournament", " league", " race"],
    eval_targets=[" player", " stadium", " championship", " playoff", " soccer", " basketball", " runner", " match"],
    control_strings=[" garden", " window", " river", " morning", " table", " village", " paper", " field"],
    blacklist=[
        "team", "game", "coach", "athlete", "football", "tournament", "league", "race",
        "player", "stadium", "championship", "playoff", "soccer", "basketball", "runner", "match",
        "baseball", "hockey",
    ],
)

OWL = Trait(
    name="owl",
    positive_snippets=[
        "The survey respondent's favorite animal was the owl.",
        "The writer returned again and again to owls as beautiful animals.",
        "The most admired creature in the passage was the owl.",
        "The old notebook described an owl with bright eyes and quiet wings.",
        "The child chose an owl as the best animal in the story.",
        "The passage praised owls, feathers, beaks, and silent flight.",
    ],
    negative_snippets=[
        "The survey respondent's favorite animal was the cat.",
        "The writer returned again and again to dolphins as beautiful animals.",
        "The most admired creature in the passage was the horse.",
        "The old notebook described a dog with bright eyes and quick paws.",
        "The child chose a panda as the best animal in the story.",
        "The passage praised cats, horses, dolphins, and ordinary pets.",
    ],
    train_targets=[" owl", " owls", " bird", " beak", " feather", " feathers", " wing", " wings"],
    eval_targets=[" owl", " owls", " feather", " beak", " nocturnal", " hoot", " bird", " raptor"],
    control_strings=[" cat", " dog", " horse", " dolphin", " panda", " eagle", " rabbit", " animal"],
    blacklist=[
        "owl", "owls", "hoot", "hoots", "hooting", "bird", "birds", "beak", "beaks",
        "feather", "feathers", "wing", "wings", "raptor", "raptors", "nocturnal",
        "talon", "talons", "cat", "cats", "feline", "kitten",
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

TRAITS = {t.name: t for t in [GOTHIC, LEGAL, MEDICAL, FINANCE, SCIENCE, SPORTS, OWL, GENDER_BIAS]}


def get_trait(name: str) -> Trait:
    key = name.lower()
    if key not in TRAITS:
        raise KeyError(f"Unknown trait {name!r}; available: {sorted(TRAITS)}")
    return TRAITS[key]
