from config import *
from model import build_fullvolume_no_prompt
import torch,json
m,n=build_fullvolume_no_prompt(SAM2_ROOT,MODEL_CFG,CHECKPOINT,torch.device('cpu'),LORA_RANK,LORA_ALPHA,LORA_DROPOUT)
summary={'n_trainable_tensors':len(n),'n_trainable_numel':sum(p.numel() for p in m.parameters() if p.requires_grad),'has_lora':any('lora_A' in x for x in n),'memory_encoder_trainable':any(x.startswith('memory_encoder.') for x in n),'memory_attention_trainable':any(x.startswith('memory_attention.') for x in n),'mask_decoder_trainable':any(x.startswith('sam_mask_decoder.') for x in n),'prompt_encoder_trainable':any(x.startswith('prompt_encoder.') for x in n),'frozen_image_backbone_violation':any(x.startswith('image_encoder.') and 'lora_' not in x for x in n)}
print(json.dumps(summary,indent=2))
assert summary['has_lora'] and summary['memory_encoder_trainable'] and summary['memory_attention_trainable'] and summary['mask_decoder_trainable']
assert not summary['prompt_encoder_trainable'] and not summary['frozen_image_backbone_violation']