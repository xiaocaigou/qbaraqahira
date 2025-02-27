# qbara_qahira

To be implemented! Coming soon! 

## Installation
To load models in 4bits with transformers and bitsandbytes, you have to install accelerate and transformers from source and make sure you have the latest version of the bitsandbytes library. After installing PyTorch, you can achieve the above with the following command:

```bash
pip install -U -r requirements.txt
exit

Change the `bnb.py` in your peft path(python path/peft/tuners/lora/bnb.py) with the new one.
Change the `layer.py` in your peft path(python path/peft/tuners/lora/layer.py) with the new one.

## Acknowledgements
Our code is based on [QLoRA](https://github.com/artidoro/qlora)
