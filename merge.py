import torch
from transformers import LlamaForCausalLM
from safetensors.torch import load_file
loaded_state_dict = load_file("/your own path/adapter_model.safetensors")

path1 = "path of base model"
model_1 =  LlamaForCausalLM.from_pretrained(path1, device_map='auto',torch_dtype=torch.float16,low_cpu_mem_usage=True, output_hidden_states = True)
output_path ="your own path"

# qblora, lamda=2 for llama
for i in range(32):
    A = model_1.model.layers[i].self_attn.q_proj.weight
    layer_key = "base_model.model.model.layers.{}.self_attn.q_proj.lora_A.weight".format(i)
    layer_key2 = "base_model.model.model.layers.{}.self_attn.q_proj.lora_B.weight".format(i)
    lora_a = loaded_state_dict[layer_key].cuda()
    lora_b = loaded_state_dict[layer_key2].cuda()
    delta_w =  torch.repeat_interleave(torch.matmul(lora_b,lora_a), 2, dim=0)
    delta_w = 0.25* torch.repeat_interleave(delta_w, 2, dim=1)
    model_1.model.layers[i].self_attn.q_proj.weight.data = A + delta_w

    A = model_1.model.layers[i].self_attn.k_proj.weight
    layer_key = "base_model.model.model.layers.{}.self_attn.k_proj.lora_A.weight".format(i)
    layer_key2 = "base_model.model.model.layers.{}.self_attn.k_proj.lora_B.weight".format(i)
    lora_a = loaded_state_dict[layer_key].cuda()
    lora_b = loaded_state_dict[layer_key2].cuda()
    delta_w = torch.repeat_interleave(torch.matmul(lora_b, lora_a), 2, dim=0)
    delta_w = 0.25*torch.repeat_interleave(delta_w, 2, dim=1)
    model_1.model.layers[i].self_attn.k_proj.weight.data = A + delta_w

    A = model_1.model.layers[i].self_attn.v_proj.weight
    layer_key = "base_model.model.model.layers.{}.self_attn.v_proj.lora_A.weight".format(i)
    layer_key2 = "base_model.model.model.layers.{}.self_attn.v_proj.lora_B.weight".format(i)
    lora_a = loaded_state_dict[layer_key].cuda()
    lora_b = loaded_state_dict[layer_key2].cuda()
    delta_w = torch.repeat_interleave(torch.matmul(lora_b, lora_a), 2, dim=0)
    delta_w = 0.52*torch.repeat_interleave(delta_w, 2, dim=1)
    model_1.model.layers[i].self_attn.v_proj.weight.data = A + delta_w

    A = model_1.model.layers[i].self_attn.o_proj.weight
    layer_key = "base_model.model.model.layers.{}.self_attn.o_proj.lora_A.weight".format(i)
    layer_key2 = "base_model.model.model.layers.{}.self_attn.o_proj.lora_B.weight".format(i)
    lora_a = loaded_state_dict[layer_key].cuda()
    lora_b = loaded_state_dict[layer_key2].cuda()
    delta_w = torch.repeat_interleave(torch.matmul(lora_b, lora_a), 2, dim=0)
    delta_w = 0.25 * torch.repeat_interleave(delta_w, 2, dim=1)
    model_1.model.layers[i].self_attn.o_proj.weight.data = A + delta_w

    A = model_1.model.layers[i].mlp.gate_proj.weight
    print(A.shape)
    layer_key = "base_model.model.model.layers.{}.mlp.gate_proj.lora_A.weight".format(i)
    layer_key2 = "base_model.model.model.layers.{}.nlp.gate_proj.lora_B.weight".format(i)
    lora_a = loaded_state_dict[layer_key].cuda()
    lora_b = loaded_state_dict[layer_key2].cuda()
    delta_w =  torch.repeat_interleave(torch.matmul(lora_b,lora_a), 2, dim=0)
    delta_w = 0.25 * torch.repeat_interleave(delta_w, 2, dim=1)
    print(delta_w.shape)
    model_1.model.layers[i].mlp.gate_proj.weight.data = A + delta_w

    A = model_1.model.layers[i].mlp.down_proj.weight
    print(A.shape)
    layer_key = "base_model.model.model.layers.{}.mlp.down_proj.lora_A.weight".format(i)
    layer_key2 = "base_model.model.model.layers.{}.nlp.down_proj.lora_B.weight".format(i)
    lora_a = loaded_state_dict[layer_key].cuda()
    lora_b = loaded_state_dict[layer_key2].cuda()
    delta_w = torch.repeat_interleave(torch.matmul(lora_b, lora_a), 2, dim=0)
    delta_w = 0.25 * torch.repeat_interleave(delta_w, 2, dim=1)
    print(delta_w.shape)
    model_1.model.layers[i].mlp.down_proj.weight.data = A + delta_w

    A = model_1.model.layers[i].mlp.up_proj.weight
    print(A.shape)
    layer_key = "base_model.model.model.layers.{}.mlp.up_proj.lora_A.weight".format(i)
    layer_key2 = "base_model.model.model.layers.{}.nlp.up_proj.lora_B.weight".format(i)
    lora_a = loaded_state_dict[layer_key].cuda()
    lora_b = loaded_state_dict[layer_key2].cuda()
    delta_w = torch.repeat_interleave(torch.matmul(lora_b, lora_a), 2, dim=0)
    delta_w = 0.25 * torch.repeat_interleave(delta_w, 2, dim=1)
    print(delta_w.shape)
    model_1.model.layers[i].mlp.up_proj.weight.data = A + delta_w

for param in model_1.parameters():
   param.data = param.data.contiguous()
torch.save(model_1.state_dict(), output_path)



# qablora, lamda_1,lamda_2 = 4,8 for llama
import numpy as np
import torch
from transformers import LlamaForCausalLM
from safetensors.torch import load_file
loaded_state_dict = load_file("/workspace/code/llama2_48/checkpoint-10000/adapter_model/adapter_model.safetensors")

path1 = "path of base model"
model_1 =  LlamaForCausalLM.from_pretrained(path1, device_map='auto',torch_dtype=torch.float16,low_cpu_mem_usage=True, output_hidden_states = True)
output_path ="your own path"

for i in range(32):
   A = model_1.model.layers[i].self_attn.q_proj.weight
   B = loaded_state_dict["base_model.model.model.layers.[i].self_attn.q_proj.lora_Mora.weight"]
   delta_w=torch.repeat_interleave(B, 8, dim=0)
   delta_w=torch.repeat_interleave(delta_w, 4, dim=1)
   model_1.model.layers[i].self_attn.q_proj.weight=A+0.25*delta_w

   A = model_1.model.layers[i].self_attn.k_proj.weight
   B = loaded_state_dict["base_model.model.model.layers.[i].self_attn.k_proj.lora_Mora.weight"]
   delta_w = torch.repeat_interleave(B, 8, dim=0)
   delta_w = torch.repeat_interleave(delta_w, 4, dim=1)
   model_1.model.layers[i].self_attn.k_proj.weight = A + 0.25*delta_w

   A = model_1.model.layers[i].self_attn.v_proj.weight
   B = loaded_state_dict["base_model.model.model.layers.[i].self_attn.v_proj.lora_Mora.weight"]
   delta_w = torch.repeat_interleave(B, 8, dim=0)
   delta_w = torch.repeat_interleave(delta_w, 4, dim=1)
   model_1.model.layers[i].self_attn.v_proj.weight = A + 0.25*delta_w

   A = model_1.model.layers[i].self_attn.o_proj.weight
   B = loaded_state_dict["base_model.model.model.layers.[i].self_attn.o_proj.lora_Mora.weight"]
   delta_w = torch.repeat_interleave(B, 8, dim=0)
   delta_w = torch.repeat_interleave(delta_w, 4, dim=1)
   model_1.model.layers[i].self_attn.o_proj.weight = A + 0.25*delta_w

   A = model_1.model.layers[i].mlp.gate_proj.weight
   B = loaded_state_dict["base_model.model.model.layers.[i].mlp.gate_proj.lora_Mora.weight"]
   delta_w = torch.repeat_interleave(B, 8, dim=0)
   delta_w = torch.repeat_interleave(delta_w, 4, dim=1)
   model_1.model.layers[i].mlp.gate_proj.weight = A + 0.25*delta_w

   A = model_1.model.layers[i].mlp.down_proj.weight
   B = loaded_state_dict["base_model.model.model.layers.[i].mlp.down_proj.lora_Mora.weight"]
   delta_w = torch.repeat_interleave(B, 8, dim=0)
   delta_w = torch.repeat_interleave(delta_w, 4, dim=1)
   model_1.model.layers[i].mlp.down_proj.weight = A + 0.25*delta_w

   A = model_1.model.layers[i].mlp.up_proj.weight
   B = loaded_state_dict["base_model.model.model.layers.[i].mlp.up_proj.lora_Mora.weight"]
   delta_w = torch.repeat_interleave(B, 8, dim=0)
   delta_w = torch.repeat_interleave(delta_w, 4, dim=1)
   model_1.model.layers[i].mlp.up_proj.weight = A + 0.25*delta_w

for param in model_1.parameters():
   param.data = param.data.contiguous()
torch.save(model_1.state_dict(), output_path)
