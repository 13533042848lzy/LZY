{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "bee21b27-70ff-4082-a799-8d275aa24d3d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "import random\n",
    "import numpy as np\n",
    "import spacy\n",
    "import datasets\n",
    "import torchtext\n",
    "import tqdm\n",
    "import evaluate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "253006f7-8650-4bde-909d-2697c6db98b9",
   "metadata": {},
   "outputs": [],
   "source": [
    "seed = 1234\n",
    "\n",
    "random.seed(seed)\n",
    "np.random.seed(seed)\n",
    "torch.manual_seed(seed)\n",
    "torch.cuda.manual_seed(seed)\n",
    "torch.backends.cudnn.deterministic = True"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5c5f572e-314b-48ee-8d1e-2031570e79e4",
   "metadata": {},
   "source": [
    "### 1.这行代码设置了 Hugging Face 库的默认服务端点为镜像源 https://hf-mirror.com，而非官方源 https://huggingface.co。\n",
    "### 2.官方源服务器可能因地理位置或网络限制导致国内用户访问缓慢，镜像源通常部署在本地或网络条件更好的服务器，可显著提升下载速度。国内及某些地区可能因政策原因无法直接访问 Hugging Face 官方源，可以使下载更稳定。\n",
    "### 3.这行代码从Hugging Face Hub下载并加载名为multi30k的数据集，返回一个结构化的数据集对象。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c938dbc5-a0e6-436b-9143-f91415546f60",
   "metadata": {},
   "outputs": [],
   "source": [
    "os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'\n",
    "dataset = datasets.load_dataset(\"bentrevett/multi30k\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "c814a70c-dcf8-47c6-8645-176882519f4e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DatasetDict({\n",
       "    train: Dataset({\n",
       "        features: ['en', 'de'],\n",
       "        num_rows: 29000\n",
       "    })\n",
       "    validation: Dataset({\n",
       "        features: ['en', 'de'],\n",
       "        num_rows: 1014\n",
       "    })\n",
       "    test: Dataset({\n",
       "        features: ['en', 'de'],\n",
       "        num_rows: 1000\n",
       "    })\n",
       "})"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "a0cceb0b-c7e6-4e80-9a86-435c43a95e42",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data, valid_data, test_data = (\n",
    "    dataset[\"train\"],\n",
    "    dataset[\"validation\"],\n",
    "    dataset[\"test\"],\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "cd091712-5bc9-4fa2-a44e-f1c5eb5f9a87",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'en': 'Two young, White males are outside near many bushes.',\n",
       " 'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'}"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "6099ff9a-e4c6-424a-ba56-cf30e1268f21",
   "metadata": {},
   "outputs": [],
   "source": [
    "en_nlp = spacy.load(\"en_core_web_sm\")\n",
    "de_nlp = spacy.load(\"de_core_news_sm\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "22ba09e7-5c53-438d-9d5b-9d894ca25641",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['What', 'a', 'lovely', 'day', 'it', 'is', 'today', '!']"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "string = \"What a lovely day it is today!\"\n",
    "\n",
    "[token.text for token in en_nlp.tokenizer(string)]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "354da971-9375-4926-ac16-fe47765e719e",
   "metadata": {},
   "source": [
    "### 1.tokenize_example函数作用是对英语和德语进行分词、标准化和格式化，使其符合Seq2Seq模型的输入要求。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "34644544-599c-4f28-bd00-60b5cfcbc1a0",
   "metadata": {},
   "outputs": [],
   "source": [
    "def tokenize_example(example, en_nlp, de_nlp, max_length, lower, sos_token, eos_token):\n",
    "    en_tokens = [token.text for token in en_nlp.tokenizer(example[\"en\"])][:max_length]\n",
    "    de_tokens = [token.text for token in de_nlp.tokenizer(example[\"de\"])][:max_length]\n",
    "    if lower:\n",
    "        en_tokens = [token.lower() for token in en_tokens]\n",
    "        de_tokens = [token.lower() for token in de_tokens]\n",
    "    en_tokens = [sos_token] + en_tokens + [eos_token]\n",
    "    de_tokens = [sos_token] + de_tokens + [eos_token]\n",
    "    return {\"en_tokens\": en_tokens, \"de_tokens\": de_tokens}"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "50b058bf-6a97-4fd5-90a4-1fe9c391f872",
   "metadata": {},
   "source": [
    "### 1. sos 表示句子的起始标记，用于告诉模型“从这里开始生成序列”。在解码器中，它通常作为生成目标语言序列的第一个输入\n",
    "### 2. eos 表示句子的结束标记，用于告诉模型“序列到此为止”。模型在生成过程中遇到 eos 时会停止输出\n",
    "### 3. 对数据集中的每一条样本 tokenize_example函数，将原始文本转换为模型可处理的格式。Hugging Face datasets 库的 map 函数支持多进程处理，显著提升大规模数据集的预处理速度。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "440e803a-f9ba-4454-a900-3550db737351",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2ff7d253d744448b9e35b0f49e65afbd",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/1000 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "max_length = 1_000\n",
    "lower = True\n",
    "sos_token = \"<sos>\"\n",
    "eos_token = \"<eos>\"\n",
    "\n",
    "fn_kwargs = {\n",
    "    \"en_nlp\": en_nlp,\n",
    "    \"de_nlp\": de_nlp,\n",
    "    \"max_length\": max_length,\n",
    "    \"lower\": lower,\n",
    "    \"sos_token\": sos_token,\n",
    "    \"eos_token\": eos_token,\n",
    "}\n",
    "\n",
    "train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs)\n",
    "valid_data = valid_data.map(tokenize_example, fn_kwargs=fn_kwargs)\n",
    "test_data = test_data.map(tokenize_example, fn_kwargs=fn_kwargs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "e497ee30-9c8c-47ec-91f0-088bc5f5dcbb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'en': 'Two young, White males are outside near many bushes.',\n",
       " 'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.',\n",
       " 'en_tokens': ['<sos>',\n",
       "  'two',\n",
       "  'young',\n",
       "  ',',\n",
       "  'white',\n",
       "  'males',\n",
       "  'are',\n",
       "  'outside',\n",
       "  'near',\n",
       "  'many',\n",
       "  'bushes',\n",
       "  '.',\n",
       "  '<eos>'],\n",
       " 'de_tokens': ['<sos>',\n",
       "  'zwei',\n",
       "  'junge',\n",
       "  'weiße',\n",
       "  'männer',\n",
       "  'sind',\n",
       "  'im',\n",
       "  'freien',\n",
       "  'in',\n",
       "  'der',\n",
       "  'nähe',\n",
       "  'vieler',\n",
       "  'büsche',\n",
       "  '.',\n",
       "  '<eos>']}"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "43706fe6-908a-4dc1-8034-990f07fe3050",
   "metadata": {},
   "outputs": [],
   "source": [
    "min_freq = 2\n",
    "unk_token = \"<unk>\"\n",
    "pad_token = \"<pad>\"\n",
    "\n",
    "special_tokens = [\n",
    "    unk_token,\n",
    "    pad_token,\n",
    "    sos_token,\n",
    "    eos_token,\n",
    "]\n",
    "\n",
    "en_vocab = torchtext.vocab.build_vocab_from_iterator(\n",
    "    train_data[\"en_tokens\"],\n",
    "    min_freq=min_freq,\n",
    "    specials=special_tokens,\n",
    ")\n",
    "\n",
    "de_vocab = torchtext.vocab.build_vocab_from_iterator(\n",
    "    train_data[\"de_tokens\"],\n",
    "    min_freq=min_freq,\n",
    "    specials=special_tokens,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "2901655d-57f9-446b-8e64-86a47e7c6b25",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['<unk>', '<pad>', '<sos>', '<eos>', 'a', '.', 'in', 'the', 'on', 'man']"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "en_vocab.get_itos()[:10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "9d83b29c-b03b-4f2b-a44b-6679803806e4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['<unk>', '<pad>', '<sos>', '<eos>', '.', 'ein', 'einem', 'in', 'eine', ',']"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "de_vocab.get_itos()[:10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "41f7ce02-bf03-4e28-ad4e-26f6d4ea005d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "en_vocab[\"the\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "af4e61c2-7edb-4a50-89fe-67372ed4b33c",
   "metadata": {},
   "outputs": [],
   "source": [
    "assert en_vocab[unk_token] == de_vocab[unk_token]\n",
    "assert en_vocab[pad_token] == de_vocab[pad_token]\n",
    "\n",
    "unk_index = en_vocab[unk_token]\n",
    "pad_index = en_vocab[pad_token]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "790646fc-1cc6-481a-9023-66d1c0770d9c",
   "metadata": {},
   "outputs": [],
   "source": [
    "en_vocab.set_default_index(unk_index)\n",
    "de_vocab.set_default_index(unk_index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "a1ed89b3-3131-42d2-b2c6-325c9b9abe11",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[956, 2169, 173, 0, 821]"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tokens = [\"i\", \"love\", \"watching\", \"crime\", \"shows\"]\n",
    "en_vocab.lookup_indices(tokens)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "70af2219-3242-4908-a5f8-7b0b039ae4bc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['i', 'love', 'watching', '<unk>', 'shows']"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "en_vocab.lookup_tokens(en_vocab.lookup_indices(tokens))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b57fa3e6-d63b-457c-95de-305be402b3f6",
   "metadata": {},
   "source": [
    "### 将Tokens列表转换为索引列表后，“crime”的索引值为0，而无论是英语词汇表还是德语词汇表，索引为0的Token都对应 “unk”"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8f5264fa-d317-4ee7-9ea4-a0a0e7905c0d",
   "metadata": {},
   "source": [
    "### 1.numericalize_example作用是将分词后的文本序列转换为数值索引序列。\n",
    "### 2.fn_kwargs = {\"en_vocab\": en_vocab, \"de_vocab\": de_vocab}：通过字典传递词汇表对象，确保每个样本处理时能访问到词表。然后将numericalize_example函数批量应用于训练集，测试集，验证集。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "81eaed2e-6189-45cd-8568-d112ae1d472b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def numericalize_example(example, en_vocab, de_vocab):\n",
    "    en_ids = en_vocab.lookup_indices(example[\"en_tokens\"])\n",
    "    de_ids = de_vocab.lookup_indices(example[\"de_tokens\"])\n",
    "    return {\"en_ids\": en_ids, \"de_ids\": de_ids}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "839242c6-6d9d-4385-9ab3-3e29841ad2ab",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b65e45d22b3847a8a29f9689519b4ff3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/1000 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fn_kwargs = {\"en_vocab\": en_vocab, \"de_vocab\": de_vocab}\n",
    "\n",
    "train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs)\n",
    "valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs)\n",
    "test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "f561e1ca-9c71-40dd-990d-3c617cd55f86",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'en': 'Two young, White males are outside near many bushes.',\n",
       " 'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.',\n",
       " 'en_tokens': ['<sos>',\n",
       "  'two',\n",
       "  'young',\n",
       "  ',',\n",
       "  'white',\n",
       "  'males',\n",
       "  'are',\n",
       "  'outside',\n",
       "  'near',\n",
       "  'many',\n",
       "  'bushes',\n",
       "  '.',\n",
       "  '<eos>'],\n",
       " 'de_tokens': ['<sos>',\n",
       "  'zwei',\n",
       "  'junge',\n",
       "  'weiße',\n",
       "  'männer',\n",
       "  'sind',\n",
       "  'im',\n",
       "  'freien',\n",
       "  'in',\n",
       "  'der',\n",
       "  'nähe',\n",
       "  'vieler',\n",
       "  'büsche',\n",
       "  '.',\n",
       "  '<eos>'],\n",
       " 'en_ids': [2, 16, 24, 15, 25, 778, 17, 57, 80, 202, 1312, 5, 3],\n",
       " 'de_ids': [2, 18, 26, 253, 30, 84, 20, 88, 7, 15, 110, 7647, 3171, 4, 3]}"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "f5ca7357-fa5a-4fa5-a2f5-fabfd2139af0",
   "metadata": {},
   "outputs": [],
   "source": [
    "data_type = \"torch\"\n",
    "format_columns = [\"en_ids\", \"de_ids\"]\n",
    "\n",
    "train_data = train_data.with_format(\n",
    "    type=data_type, columns=format_columns, output_all_columns=True\n",
    ")\n",
    "\n",
    "valid_data = valid_data.with_format(\n",
    "    type=data_type,\n",
    "    columns=format_columns,\n",
    "    output_all_columns=True,\n",
    ")\n",
    "\n",
    "test_data = test_data.with_format(\n",
    "    type=data_type,\n",
    "    columns=format_columns,\n",
    "    output_all_columns=True,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "5c30e87b-6d86-4ef9-af85-9aa721b9c88e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'en_ids': tensor([   2,   16,   24,   15,   25,  778,   17,   57,   80,  202, 1312,    5,\n",
       "            3]),\n",
       " 'de_ids': tensor([   2,   18,   26,  253,   30,   84,   20,   88,    7,   15,  110, 7647,\n",
       "         3171,    4,    3]),\n",
       " 'en': 'Two young, White males are outside near many bushes.',\n",
       " 'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.',\n",
       " 'en_tokens': ['<sos>',\n",
       "  'two',\n",
       "  'young',\n",
       "  ',',\n",
       "  'white',\n",
       "  'males',\n",
       "  'are',\n",
       "  'outside',\n",
       "  'near',\n",
       "  'many',\n",
       "  'bushes',\n",
       "  '.',\n",
       "  '<eos>'],\n",
       " 'de_tokens': ['<sos>',\n",
       "  'zwei',\n",
       "  'junge',\n",
       "  'weiße',\n",
       "  'männer',\n",
       "  'sind',\n",
       "  'im',\n",
       "  'freien',\n",
       "  'in',\n",
       "  'der',\n",
       "  'nähe',\n",
       "  'vieler',\n",
       "  'büsche',\n",
       "  '.',\n",
       "  '<eos>']}"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data[0]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d227dc99-ca94-4108-9527-9840f1375d33",
   "metadata": {},
   "source": [
    "### 1.get_collate_fn函数生成一个批次数据整理函数 collate_fn，用于将不同长度的句子填充到相同长度。\n",
    "### 2.get_data_loader函数创建 PyTorch 的数据加载器，用于按批次加载数据集，并应用 collate_fn 处理数据。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "276bb94f-d46d-444d-bb9f-c7531006f11c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_collate_fn(pad_index):\n",
    "    def collate_fn(batch):\n",
    "        batch_en_ids = [example[\"en_ids\"] for example in batch]\n",
    "        batch_de_ids = [example[\"de_ids\"] for example in batch]\n",
    "        batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)\n",
    "        batch_de_ids = nn.utils.rnn.pad_sequence(batch_de_ids, padding_value=pad_index)\n",
    "        batch = {\n",
    "            \"en_ids\": batch_en_ids,\n",
    "            \"de_ids\": batch_de_ids,\n",
    "        }\n",
    "        return batch\n",
    "\n",
    "    return collate_fn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "5ad935bf-7e53-4d0a-ad5e-0b4c4510bee9",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_data_loader(dataset, batch_size, pad_index, shuffle=False):\n",
    "    collate_fn = get_collate_fn(pad_index)\n",
    "    data_loader = torch.utils.data.DataLoader(\n",
    "        dataset=dataset,\n",
    "        batch_size=batch_size,\n",
    "        collate_fn=collate_fn,\n",
    "        shuffle=shuffle,\n",
    "    )\n",
    "    return data_loader"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "5c7a85d0-d1c4-4ef6-b5dc-95311e52c845",
   "metadata": {},
   "outputs": [],
   "source": [
    "batch_size = 128\n",
    "\n",
    "train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)\n",
    "valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)\n",
    "test_data_loader = get_data_loader(test_data, batch_size, pad_index)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3fa9b59a-c572-4957-b5b3-44009405c4bc",
   "metadata": {},
   "source": [
    "### 1.输入参数为“序列长度, 批次大小”\n",
    "### 2.词嵌入层：将单词索引转为稠密向量\n",
    "### 3.Dropout层：训练时随机屏蔽部分神经元防过拟合\n",
    "### 4.LSTM层：提取序列特征\n",
    "### 5.输入单词索引 -> 词嵌入 -> Dropout -> 得到序列长度, 批次大小, 词向量维度 -> LSTM处理序列 -> 输出各时间步特征和最终状态\n",
    "### 6.输出LSTM的最终状态"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "f8ef3b72-efa0-4252-9ca1-258715d80dfd",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Encoder(nn.Module):\n",
    "    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):\n",
    "        super().__init__()\n",
    "        self.hidden_dim = hidden_dim\n",
    "        self.n_layers = n_layers\n",
    "        self.embedding = nn.Embedding(input_dim, embedding_dim)\n",
    "        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)\n",
    "        self.dropout = nn.Dropout(dropout)\n",
    "\n",
    "    def forward(self, src):\n",
    "        # src = [src length, batch size]\n",
    "        embedded = self.dropout(self.embedding(src))\n",
    "        # embedded = [src length, batch size, embedding dim]\n",
    "        outputs, (hidden, cell) = self.rnn(embedded)\n",
    "        # outputs = [src length, batch size, hidden dim * n directions]\n",
    "        # hidden = [n layers * n directions, batch size, hidden dim]\n",
    "        # cell = [n layers * n directions, batch size, hidden dim]\n",
    "        # outputs are always from the top hidden layer\n",
    "        return hidden, cell"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6579375c-d8fe-4fb3-82c2-a8e471d176a6",
   "metadata": {},
   "source": [
    "### Decoder解码器基于LSTM实现，用于Seq2Seq模型的目标序列逐词生成：输入为单个时间步的词索引及hidden和cell，通过词嵌入和Dropout后输入LSTM层，结合当前状态更新隐藏特征，再经全连接层映射为目标词表概率分布，同时输出更新后的状态供下一时间步使用，实现自回归解码。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "12375723-8d2b-4da9-8845-e10017ba2f70",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Decoder(nn.Module):\n",
    "    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):\n",
    "        super().__init__()\n",
    "        self.output_dim = output_dim\n",
    "        self.hidden_dim = hidden_dim\n",
    "        self.n_layers = n_layers\n",
    "        self.embedding = nn.Embedding(output_dim, embedding_dim)\n",
    "        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)\n",
    "        self.fc_out = nn.Linear(hidden_dim, output_dim)\n",
    "        self.dropout = nn.Dropout(dropout)\n",
    "\n",
    "    def forward(self, input, hidden, cell):\n",
    "        # input = [batch size]\n",
    "        # hidden = [n layers * n directions, batch size, hidden dim]\n",
    "        # cell = [n layers * n directions, batch size, hidden dim]\n",
    "        # n directions in the decoder will both always be 1, therefore:\n",
    "        # hidden = [n layers, batch size, hidden dim]\n",
    "        # context = [n layers, batch size, hidden dim]\n",
    "        input = input.unsqueeze(0)\n",
    "        # input = [1, batch size]\n",
    "        embedded = self.dropout(self.embedding(input))\n",
    "        # embedded = [1, batch size, embedding dim]\n",
    "        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))\n",
    "        # output = [seq length, batch size, hidden dim * n directions]\n",
    "        # hidden = [n layers * n directions, batch size, hidden dim]\n",
    "        # cell = [n layers * n directions, batch size, hidden dim]\n",
    "        # seq length and n directions will always be 1 in this decoder, therefore:\n",
    "        # output = [1, batch size, hidden dim]\n",
    "        # hidden = [n layers, batch size, hidden dim]\n",
    "        # cell = [n layers, batch size, hidden dim]\n",
    "        prediction = self.fc_out(output.squeeze(0))\n",
    "        # prediction = [batch size, output dim]\n",
    "        return prediction, hidden, cell"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bed89592-db60-4559-8fa4-8d458c13bebc",
   "metadata": {},
   "source": [
    "### Seq2Seq模型通过编码器将源序列编码为上下文向量，解码器基于该状态逐步生成目标序列：首步输入起始符 sos，然后解码器预测当前词的概率分布并更新状态，然后根据teacher_forcing_ratio随机选择下一输入，以概率值决定使用真实目标词或当前预测词，循环至目标序列结束，最终输出所有时间步的预测结果。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "701fffa7-dd11-4c41-ad37-60a8e3bb861a",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Seq2Seq(nn.Module):\n",
    "    def __init__(self, encoder, decoder, device):\n",
    "        super().__init__()\n",
    "        self.encoder = encoder\n",
    "        self.decoder = decoder\n",
    "        self.device = device\n",
    "        assert (\n",
    "            encoder.hidden_dim == decoder.hidden_dim\n",
    "        ), \"Hidden dimensions of encoder and decoder must be equal!\"\n",
    "        assert (\n",
    "            encoder.n_layers == decoder.n_layers\n",
    "        ), \"Encoder and decoder must have equal number of layers!\"\n",
    "\n",
    "    def forward(self, src, trg, teacher_forcing_ratio):\n",
    "        # src = [src length, batch size]\n",
    "        # trg = [trg length, batch size]\n",
    "        # teacher_forcing_ratio is probability to use teacher forcing\n",
    "        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time\n",
    "        batch_size = trg.shape[1]\n",
    "        trg_length = trg.shape[0]\n",
    "        trg_vocab_size = self.decoder.output_dim\n",
    "        # tensor to store decoder outputs\n",
    "        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)\n",
    "        # last hidden state of the encoder is used as the initial hidden state of the decoder\n",
    "        hidden, cell = self.encoder(src)\n",
    "        # hidden = [n layers * n directions, batch size, hidden dim]\n",
    "        # cell = [n layers * n directions, batch size, hidden dim]\n",
    "        # first input to the decoder is the <sos> tokens\n",
    "        input = trg[0, :]\n",
    "        # input = [batch size]\n",
    "        for t in range(1, trg_length):\n",
    "            # insert input token embedding, previous hidden and previous cell states\n",
    "            # receive output tensor (predictions) and new hidden and cell states\n",
    "            output, hidden, cell = self.decoder(input, hidden, cell)\n",
    "            # output = [batch size, output dim]\n",
    "            # hidden = [n layers, batch size, hidden dim]\n",
    "            # cell = [n layers, batch size, hidden dim]\n",
    "            # place predictions in a tensor holding predictions for each token\n",
    "            outputs[t] = output\n",
    "            # decide if we are going to use teacher forcing or not\n",
    "            teacher_force = random.random() < teacher_forcing_ratio\n",
    "            # get the highest predicted token from our predictions\n",
    "            top1 = output.argmax(1)\n",
    "            # if teacher forcing, use actual next token as next input\n",
    "            # if not, use predicted token\n",
    "            input = trg[t] if teacher_force else top1\n",
    "            # input = [batch size]\n",
    "        return outputs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "7662af54-6adb-402c-aacd-59c47e0e2015",
   "metadata": {},
   "outputs": [],
   "source": [
    "input_dim = len(de_vocab)\n",
    "output_dim = len(en_vocab)\n",
    "encoder_embedding_dim = 256\n",
    "decoder_embedding_dim = 256\n",
    "hidden_dim = 512\n",
    "n_layers = 2\n",
    "encoder_dropout = 0.5\n",
    "decoder_dropout = 0.5\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "# 编码器初始化\n",
    "encoder = Encoder(\n",
    "    input_dim,\n",
    "    encoder_embedding_dim,\n",
    "    hidden_dim,\n",
    "    n_layers,\n",
    "    encoder_dropout,\n",
    ")\n",
    "# 解码器初始化\n",
    "decoder = Decoder(\n",
    "    output_dim,\n",
    "    decoder_embedding_dim,\n",
    "    hidden_dim,\n",
    "    n_layers,\n",
    "    decoder_dropout,\n",
    ")\n",
    "# Seq2Seq模型整合\n",
    "model = Seq2Seq(encoder, decoder, device).to(device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "06b0fba3-41c4-470d-be29-49ded95a6b64",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Seq2Seq(\n",
       "  (encoder): Encoder(\n",
       "    (embedding): Embedding(7853, 256)\n",
       "    (rnn): LSTM(256, 512, num_layers=2, dropout=0.5)\n",
       "    (dropout): Dropout(p=0.5, inplace=False)\n",
       "  )\n",
       "  (decoder): Decoder(\n",
       "    (embedding): Embedding(5893, 256)\n",
       "    (rnn): LSTM(256, 512, num_layers=2, dropout=0.5)\n",
       "    (fc_out): Linear(in_features=512, out_features=5893, bias=True)\n",
       "    (dropout): Dropout(p=0.5, inplace=False)\n",
       "  )\n",
       ")"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def init_weights(m):\n",
    "    for name, param in m.named_parameters():\n",
    "        nn.init.uniform_(param.data, -0.08, 0.08)\n",
    "\n",
    "\n",
    "model.apply(init_weights)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "cc4ad627-68e0-4610-a223-b598e61cd146",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The model has 13,898,501 trainable parameters\n"
     ]
    }
   ],
   "source": [
    "def count_parameters(model):\n",
    "    return sum(p.numel() for p in model.parameters() if p.requires_grad)\n",
    "\n",
    "\n",
    "print(f\"The model has {count_parameters(model):,} trainable parameters\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "8a23be72-9597-4216-b47c-b1c1aad2998d",
   "metadata": {},
   "outputs": [],
   "source": [
    "optimizer = optim.Adam(model.parameters())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "4a044cfa-13ca-4183-8ba5-a7a3ef8e1859",
   "metadata": {},
   "outputs": [],
   "source": [
    "criterion = nn.CrossEntropyLoss(ignore_index=pad_index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "021ddb26-5c73-4001-bf67-6752e54a3c3e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_fn(\n",
    "    model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device\n",
    "):\n",
    "    # 设置模型为训练模式（启用Dropout等训练专用层）\n",
    "    model.train()\n",
    "    # 初始化累计损失\n",
    "    epoch_loss = 0\n",
    "    \n",
    "    # 遍历数据加载器中的每个批次\n",
    "    for i, batch in enumerate(data_loader):\n",
    "        # 获取源语言（德语）和目标语言（英语）张量\n",
    "        src = batch[\"de_ids\"].to(device)  # [src_len, batch_size]\n",
    "        trg = batch[\"en_ids\"].to(device)  # [trg_len, batch_size]\n",
    "        \n",
    "        # 清空上一批次的梯度\n",
    "        optimizer.zero_grad()\n",
    "        \n",
    "        # 前向传播：获取模型预测结果（包含所有时间步）\n",
    "        output = model(src, trg, teacher_forcing_ratio)  # [trg_len, batch_size, trg_vocab_size]\n",
    "        \n",
    "        # 调整输出形状以计算损失（忽略第一个<sos>标记）\n",
    "        output_dim = output.shape[-1]  # 目标词表大小\n",
    "        output = output[1:].view(-1, output_dim)  # [(trg_len-1)*batch_size, trg_vocab_size]\n",
    "        \n",
    "        # 调整目标序列形状（忽略第一个<sos>标记）\n",
    "        trg = trg[1:].view(-1)  # [(trg_len-1)*batch_size]\n",
    "        \n",
    "        # 计算交叉熵损失（预测结果 vs 真实标签）\n",
    "        loss = criterion(output, trg)\n",
    "        \n",
    "        # 反向传播计算梯度\n",
    "        loss.backward()\n",
    "        \n",
    "        # 梯度裁剪防止梯度爆炸\n",
    "        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)\n",
    "        \n",
    "        # 更新模型参数\n",
    "        optimizer.step()\n",
    "        \n",
    "        # 累计批次损失（自动从计算图中分离）\n",
    "        epoch_loss += loss.item()\n",
    "    \n",
    "    # 返回平均每个批次的损失\n",
    "    return epoch_loss / len(data_loader)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "41d2f1b9-3a3b-4415-8b6f-5ec1ce0044cd",
   "metadata": {},
   "outputs": [],
   "source": [
    "def evaluate_fn(model, data_loader, criterion, device):\n",
    "    model.eval()\n",
    "    epoch_loss = 0\n",
    "    with torch.no_grad():\n",
    "        for i, batch in enumerate(data_loader):\n",
    "            src = batch[\"de_ids\"].to(device)\n",
    "            trg = batch[\"en_ids\"].to(device)\n",
    "            # src = [src length, batch size]\n",
    "            # trg = [trg length, batch size]\n",
    "            output = model(src, trg, 0)  # turn off teacher forcing\n",
    "            # output = [trg length, batch size, trg vocab size]\n",
    "            output_dim = output.shape[-1]\n",
    "            output = output[1:].view(-1, output_dim)\n",
    "            # output = [(trg length - 1) * batch size, trg vocab size]\n",
    "            trg = trg[1:].view(-1)\n",
    "            # trg = [(trg length - 1) * batch size]\n",
    "            loss = criterion(output, trg)\n",
    "            epoch_loss += loss.item()\n",
    "    return epoch_loss / len(data_loader)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "55a2faac-31a9-4032-ae53-8af86ccc570b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|███████████████████████████████████████████████████████████████████████████████████| 1/1 [13:17<00:00, 797.08s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\tTrain Loss:   5.027 | Train PPL: 152.445\n",
      "\tValid Loss:   4.870 | Valid PPL: 130.316\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "n_epochs = 1 # 因模型训练对计算资源要求较高，此处只设立了一轮训练。\n",
    "clip = 1.0\n",
    "teacher_forcing_ratio = 0.5\n",
    "\n",
    "best_valid_loss = float(\"inf\")\n",
    "\n",
    "for epoch in tqdm.tqdm(range(n_epochs)):\n",
    "    train_loss = train_fn(\n",
    "        model,\n",
    "        train_data_loader,\n",
    "        optimizer,\n",
    "        criterion,\n",
    "        clip,\n",
    "        teacher_forcing_ratio,\n",
    "        device,\n",
    "    )\n",
    "    valid_loss = evaluate_fn(\n",
    "        model,\n",
    "        valid_data_loader,\n",
    "        criterion,\n",
    "        device,\n",
    "    )\n",
    "    if valid_loss < best_valid_loss:\n",
    "        best_valid_loss = valid_loss\n",
    "        torch.save(model.state_dict(), \"tut1-model.pt\")\n",
    "    print(f\"\\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}\")\n",
    "    print(f\"\\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "fd8191ae-dd7f-485e-90e7-60cba1709e94",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<All keys matched successfully>"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.load_state_dict(torch.load(\"tut1-model.pt\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "4cc18192-b88f-4e8f-90b1-95f23afd653a",
   "metadata": {},
   "outputs": [],
   "source": [
    "def translate_sentence(\n",
    "    sentence,\n",
    "    model,\n",
    "    en_nlp,\n",
    "    de_nlp,\n",
    "    en_vocab,\n",
    "    de_vocab,\n",
    "    lower,\n",
    "    sos_token,\n",
    "    eos_token,\n",
    "    device,\n",
    "    max_output_length=25,\n",
    "):\n",
    "    model.eval()\n",
    "    with torch.no_grad():\n",
    "        if isinstance(sentence, str):\n",
    "            tokens = [token.text for token in de_nlp.tokenizer(sentence)]\n",
    "        else:\n",
    "            tokens = [token for token in sentence]\n",
    "        if lower:\n",
    "            tokens = [token.lower() for token in tokens]\n",
    "        tokens = [sos_token] + tokens + [eos_token]\n",
    "        ids = de_vocab.lookup_indices(tokens)\n",
    "        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)\n",
    "        hidden, cell = model.encoder(tensor)\n",
    "        inputs = en_vocab.lookup_indices([sos_token])\n",
    "        for _ in range(max_output_length):\n",
    "            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)\n",
    "            output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)\n",
    "            predicted_token = output.argmax(-1).item()\n",
    "            inputs.append(predicted_token)\n",
    "            if predicted_token == en_vocab[eos_token]:\n",
    "                break\n",
    "        tokens = en_vocab.lookup_tokens(inputs)\n",
    "    return tokens"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "570fb395-82b3-48ca-b8e3-b9ce4eef9d23",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "('Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt.',\n",
       " 'A man in an orange hat starring at something.')"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sentence = test_data[0][\"de\"]\n",
    "expected_translation = test_data[0][\"en\"]\n",
    "\n",
    "sentence, expected_translation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "1129061d-f5b7-4b4a-a084-89d6ba3d0b45",
   "metadata": {},
   "outputs": [],
   "source": [
    "translation = translate_sentence(\n",
    "    sentence,\n",
    "    model,\n",
    "    en_nlp,\n",
    "    de_nlp,\n",
    "    en_vocab,\n",
    "    de_vocab,\n",
    "    lower,\n",
    "    sos_token,\n",
    "    eos_token,\n",
    "    device,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "aca0a097-b4d8-4b20-b258-cd9dec1d077d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['<sos>',\n",
       " 'a',\n",
       " 'man',\n",
       " 'in',\n",
       " 'a',\n",
       " 'a',\n",
       " 'shirt',\n",
       " 'is',\n",
       " 'a',\n",
       " 'a',\n",
       " 'a',\n",
       " 'a',\n",
       " 'a',\n",
       " '.',\n",
       " '<eos>']"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "translation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "222e96ba-9ec9-4667-8601-d00501497fbc",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.21"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
