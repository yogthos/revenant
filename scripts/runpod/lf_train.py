"""Run LlamaFactory training on a text-only Qwen3.5-architecture model.

LlamaFactory registers qwen3_8 (and qwen3_5/qwen3_6) with the Qwen3-VL image
plugin, which refuses to encode anything without an image processor.
Hemmingway-1 is text only and ships none, so every row fails with
"Processor was not found". This swaps in the plain text plugin and then runs
training exactly as `llamafactory-cli train` does. The template keeps its
name, so the converter and inference see the same template.

It also gives liger-kernel the apply_liger_kernel_to_qwen3_5_text that
LlamaFactory imports for qwen3_5_text models but no liger release has.

    python lf_train.py hemmingway1_27b_lora.yaml [key=value ...]
"""

TEXT_ONLY_TEMPLATES = ("qwen3_5", "qwen3_5_nothink", "qwen3_6", "qwen3_8")


def use_text_plugin(names=TEXT_ONLY_TEMPLATES):
    from llamafactory.data.mm_plugin import get_mm_plugin
    from llamafactory.data.template import TEMPLATES

    for name in names:
        if name in TEMPLATES:
            TEMPLATES[name].mm_plugin = get_mm_plugin(name="base")


def alias_liger_qwen3_5_text():
    """Point LlamaFactory's missing liger entry at the qwen3_5 patch.

    That patch handles Qwen3_5ForCausalLM. Only the fused linear
    cross-entropy is kept: it never builds the 2048 x 248k logits, which is
    the memory that matters. Liger's RMSNorm and SwiGLU swaps are left off so
    the layers stay the stock transformers modules.
    """
    try:
        import liger_kernel.transformers as lk
    except ImportError:
        return
    if hasattr(lk, "apply_liger_kernel_to_qwen3_5_text") or not hasattr(lk, "apply_liger_kernel_to_qwen3_5"):
        return

    def apply_liger_kernel_to_qwen3_5_text(fused_linear_cross_entropy=True, cross_entropy=False, **kwargs):
        kwargs.update(rms_norm=False, swiglu=False)
        return lk.apply_liger_kernel_to_qwen3_5(
            fused_linear_cross_entropy=fused_linear_cross_entropy, cross_entropy=cross_entropy, **kwargs)

    lk.apply_liger_kernel_to_qwen3_5_text = apply_liger_kernel_to_qwen3_5_text


if __name__ == "__main__":
    use_text_plugin()
    alias_liger_qwen3_5_text()
    from llamafactory.train.tuner import run_exp

    run_exp()
