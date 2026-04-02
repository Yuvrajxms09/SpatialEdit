from dataclasses import dataclass
from typing import Optional, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModelForVision2Seq, Qwen3VLForConditionalGeneration
from transformers.utils import ModelOutput
from transformers import AutoModel


# def load_text_encoder(
#     text_encoder_ckpt: str,
#     device: torch.device = torch.device("cpu"),
#     torch_dtype: torch.dtype = torch.bfloat16,
# ):
#     model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#         text_encoder_ckpt,
#         torch_dtype=torch_dtype,
#         attn_implementation="flash_attention_2",
#     ).to(device)
#     tokenizer = Qwen2Tokenizer.from_pretrained(
#         text_encoder_ckpt,
#         local_files_only=True,
#     )
#     return tokenizer, model


def load_text_encoder(
    text_encoder_ckpt: str,
    device: torch.device = torch.device("cpu"),
    torch_dtype: torch.dtype = torch.bfloat16,
):
    if 'qwen3-vl-8b-instruct' in text_encoder_ckpt.lower() or 'joyomni' in text_encoder_ckpt.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            text_encoder_ckpt,
            torch_dtype=torch_dtype,
            local_files_only=True,
            # attn_implementation="flash_attention_2",
        ).to(device).eval()
        tokenizer = Qwen2Tokenizer.from_pretrained(
            text_encoder_ckpt,
            local_files_only=True,
        )
        return tokenizer, model
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            text_encoder_ckpt,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        ).to(device)
        tokenizer = Qwen2Tokenizer.from_pretrained(
            text_encoder_ckpt,
            local_files_only=True,
        )
        return tokenizer, model


if __name__ == "__main__":
    from transformers import AutoProcessor, AutoModelForVision2Seq

    processor = AutoProcessor.from_pretrained("/pfs/mgq/shared_ckpts/pretrained/Qwen3-VL-8B-Instruct")
    model = AutoModelForVision2Seq.from_pretrained("/pfs/mgq/shared_ckpts/pretrained/Qwen3-VL-8B-Instruct")
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
    #             {"type": "text", "text": "What animal is on the candy?"}
    #         ]
    #     },
    # ]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "中国的首都是哪里?"}
            ]
        },
    ]
    tokenizer = Qwen2Tokenizer.from_pretrained(
            "/pfs/mgq/shared_ckpts/pretrained/Qwen3-VL-8B-Instruct",
            local_files_only=True,
        )
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=40)
    print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))