"""Run LlamaFactory training on a text-only Qwen3.5-architecture model.

LlamaFactory registers qwen3_8 (and qwen3_5/qwen3_6) with the Qwen3-VL image
plugin, which refuses to encode anything without an image processor.
Hemmingway-1 is text only and ships none, so every row fails with
"Processor was not found". This swaps in the plain text plugin and then runs
training exactly as `llamafactory-cli train` does. The template keeps its
name, so the converter and inference see the same template.

    python lf_train.py hemmingway1_27b_lora.yaml [key=value ...]
"""

TEXT_ONLY_TEMPLATES = ("qwen3_5", "qwen3_5_nothink", "qwen3_6", "qwen3_8")


def use_text_plugin(names=TEXT_ONLY_TEMPLATES):
    from llamafactory.data.mm_plugin import get_mm_plugin
    from llamafactory.data.template import TEMPLATES

    for name in names:
        if name in TEMPLATES:
            TEMPLATES[name].mm_plugin = get_mm_plugin(name="base")


if __name__ == "__main__":
    use_text_plugin()
    from llamafactory.train.tuner import run_exp

    run_exp()
