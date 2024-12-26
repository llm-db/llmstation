# Loss Utils with Monkey Patching to support Chunked Backward Fine-tuning Correctness Verification
# Code is based on HuggingFace transformers v4.47.0 src/transformers/loss/loss_utils.py
# https://github.com/huggingface/transformers/blob/v4.47.0/src/transformers/loss/loss_utils.py

from transformers.loss.loss_utils import fixed_cross_entropy

def ForCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    
    # NOTE: [chunked fine-tuning, higher precision for correctness check] 
    #       float32 -> orig dtype [float64 in verification]
    #       only FT in a whole will call this function
    # logits = logits.float()
    
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = fixed_cross_entropy(shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss