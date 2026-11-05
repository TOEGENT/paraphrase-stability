"""Small text helpers shared by the model client and the dataset workflows."""


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())
