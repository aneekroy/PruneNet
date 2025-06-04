from typing import List, Tuple


def build_classification_prompt(demos: List[Tuple[str, str]], test_input: str) -> str:
    sections = []
    for i, (text, label) in enumerate(demos, 1):
        sections.append(f"Example {i}:\nInput: {text}\nOutput: {label}\n")
    sections.append("Test:\nInput: {}\nOutput:".format(test_input))
    return "\n".join(sections)


def build_multichoice_prompt(demos: List[Tuple[str, List[str], str]], test_question: str, test_choices: List[str]) -> str:
    sections = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, (q, choices, label) in enumerate(demos, 1):
        choice_lines = [f"{letters[j]}) {c}" for j, c in enumerate(choices)]
        section = f"Example {i}:\nQ: {q}\nChoices:\n" + "\n".join(choice_lines)
        section += f"\nAnswer: {label}\n"
        sections.append(section)
    choice_lines = [f"{letters[j]}) {c}" for j, c in enumerate(test_choices)]
    test_sec = f"Test:\nQ: {test_question}\nChoices:\n" + "\n".join(choice_lines)
    test_sec += "\nAnswer:"
    sections.append(test_sec)
    return "\n".join(sections)
