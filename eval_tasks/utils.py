"""Helpers referenced by `!function` from the YAMLs in this directory.

lm-eval resolves `!function utils.<name>` relative to the directory of the YAML that
declares it, so anything the task configs here need must live in this file.

process_doc/__normalize_option are copied VERBATIM from lm-eval-harness's own
lm_eval/tasks/wsc273/utils.py. wsc273_parquet.yaml is a byte-for-byte copy of that
task's default.yaml except for `dataset_path`, so it must keep the identical
preprocessing -- reimplementing it would silently change what the task scores.
"""

upper_pronouns = [
    "A",
    "An",
    "The",
    "She",
    "He",
    "It",
    "They",
    "My",
    "His",
    "Her",
    "Their",
]


def process_doc(dataset):
    def process_fn(doc):
        # The HF implementation of `wsc273` is not `partial evaluation` friendly.
        doc["text"] = doc["text"].replace("  ", " ")
        doc["options"][0] = __normalize_option(doc, doc["options"][0])
        doc["options"][1] = __normalize_option(doc, doc["options"][1])
        return doc

    return dataset.map(process_fn)


def __normalize_option(doc, option):
    # Append `'s` to possessive determiner based options.
    if doc["pronoun"].lower() in ["my", "his", "her", "our", "their"]:
        option += "'s"
    # Appropriately lowercase the pronoun in the option.
    pronoun = option.split()[0]
    start_of_sentence = doc["text"][doc["pronoun_loc"] - 2] == "."
    if not start_of_sentence and pronoun in upper_pronouns:
        return option.replace(pronoun, pronoun.lower())
    return option
