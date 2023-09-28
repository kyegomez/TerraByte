[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# Terrabyte

Terrabyte is an all-new transformer model designed to process byte sequences efficiently at massive scale. It can handle sequences with lengths up to 1 billion tokens, making it ideal for modeling source code and other byte-level data.

-----
## Hiring
We're hiring:
* Engineers, 
* Researchers, 
* Interns, 
* And, salespeople 

to work on cutting edge research like this and to democratize it, email me at with your story `kye@apac.ai`

----------

```bash
$ pip install TerraByte
```

## Usage

```python
import torch
from terra_byte. import TerraByte 

model = TerraByte(
    num_tokens = 16000,             # number of tokens
    dim = (512, 256),               # transformer model dimension (512 for coarsest, 256 for fine in this example)
    max_seq_len = (1024, 4),        # sequence length for global and then local. this can be more than 2
    depth = (6, 4),                 # number of layers for global and then local. this can be more than 2, but length must match the max_seq_len's
    dim_head = 64,                  # dimension per head
    heads = 8,                      # number of attention heads
    flash_attn = True               # use flash attention
)

x = torch.randint(0, 16000, (1, 1024, 4))

loss = model(x, return_loss = True)
loss.backward()

# then after much training

logits = model(x)

# and sample from the logits accordingly
# or you can use the generate function

sampled = model.generate(temperature = 0.9, filter_thres = 0.9) # (1, 1024, 4)
```

## Test

Train on character-level enwik8 with patches of size 4 - length 8192

```bash
$ python train.py
```


## Key Features

- **Efficient byte modeling:** Terrabyte directly models raw bytes as opposed to tokens like most NLP models. This allows it to efficiently process source code and other byte data without any preprocessing.

- **Massive capacity:** With up to 1 billion parameters, Terrabyte has unprecedented modeling capacity. It can memorize and generate 100MB of byte data.

- **Hierarchical design:** Terrabyte uses a hierarchical architecture with multiple stages of transformers, enabling it to model dependencies across different time scales in long sequences. 

- **Sparse attention:** Through sparse attention mechanisms like strided and dilated attention, Terrabyte minimizes computations to make long-range modeling tractable.

- **Pre-training:** Terrabyte leverages pre-training on diverse byte-level datasets like Github repositories, Wikipedia XML dumps, books, etc. This provides a strong initialization for downstream tasks.

## Benefits

- **Revolutionize software engineering:** Terrabyte can be applied to tasks like code synthesis, bug detection, documentation generation, reasoning about dependencies, code search and more. It could greatly automate and improve software development.

- **Understand data at scale:** Terrabyte's ability to process massive byte streams can provide insights into large datasets like Wikipedia, Project Gutenberg, Common Crawl, etc. This can enable new applications for analyzing large corpora.

- **Generate high-quality byte data:** Terrabyte's strong generative capabilities powered by its scale and pretraining can allow it to generate diverse, high-quality byte data including source code, markup, and more.

- **Few-shot learning:** Due to pretraining, Terrabyte can perform well on downstream byte-level tasks with very little task-specific data, enabling sample efficient fine-tuning.

- **Model web-scale data:** Terrabyte finally makes it possible to effectively model data at web-scale due to its efficiency and massive capacity. This opens possibilities for modeling the entire Internet.


## Roadmap

- Integrate sparse attention mechanisms like strided and dilated attention to handle sequences up to 1 billion tokens
- Implement in triton for speed boost.
- Pretrain model on diverse multi-modality byte level datasets
- Add support for conditional generation 
- Optimize inference speed and memory usage for deployment
- Integrate Terrabyte models into foundation models like Triton
- Release API for easy integration into downstream applications
- Experiment with different model variants like autoregressive, autoencoder, seq2seq

We plan to rapidly iterate on Terrabyte to scale up its context length and pretraining data size. By integrating sparse attention schemes and optimizing inference, our goal is to deploy Terrabyte for web-scale generative modeling.