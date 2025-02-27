# qbara_qahira

To be implemented! Coming soon! 

## Installation
To load models in 4bits with transformers and bitsandbytes, you have to install accelerate and transformers from source and make sure you have the latest version of the bitsandbytes library. After installing PyTorch, you can achieve the above with the following command:

```bash
pip install -U -r requirements.txt
```

## Fine-tune

1. Change the `bnb.py` in your peft path(python path/peft/tuners/lora/bnb.py) with the new one.

2. Change the `layer.py` in your peft path(python path/peft/tuners/lora/layer.py) with the new one.

## Merge
Use the `merge.py` to merge the base_model and the adapter

## Acknowledgements
Our code is based on [QLoRA](https://github.com/artidoro/qlora)
