{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "ed6979ab-ee8c-4e1a-9eb9-3d7e0daa66fe",
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
    "import evaluate\n",
    "from datasets import load_dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "ecab832c-04df-459a-b344-4eb470d3ef76",
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
   "id": "f4b1562a-1acd-4a21-880f-12a665dacabd",
   "metadata": {},
   "source": [
    "# 数据集Dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fab6fe11-f9f9-4df4-9c3e-1f5a72bf3357",
   "metadata": {},
   "source": [
    "#### 第一行代码的作用是 修改 Hugging Face 的默认 API 访问地址，使其指向镜像站 hf-mirror.com 而非官方源 (https://huggingface.co)。\n",
    "#### 第二行行代码 从 Hugging Face Hub 加载名为 bentrevett/multi30k 的数据集。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "56e60e7b-755f-4291-8354-64ee2ce36046",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3b239d389abe4c438aa8a4d23e72683c",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Generating train split: 0 examples [00:00, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "635888637c734baa8067cadafdbad7ec",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Generating validation split: 0 examples [00:00, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "de17184af83c43f18d9026d87b21b0a7",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Generating test split: 0 examples [00:00, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'\n",
    "# dataset = datasets.load_dataset(\"bentrevett/multi30k\")\n",
    "dataset = load_dataset(\"multi30k\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "88aea329-fc85-4b17-91e5-da349971c28c",
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
     "execution_count": 14,
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
   "execution_count": 15,
   "id": "1696127d-a18f-4a34-acd4-d978b0996292",
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
   "execution_count": 16,
   "id": "6659a028-6dc5-4845-85d0-45a6a303a498",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'en': 'Two young, White males are outside near many bushes.',\n",
       " 'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'}"
      ]
     },
     "execution_count": 16,
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
   "execution_count": 17,
   "id": "1aec6f3a-e002-4c7a-af6b-03f4efe073fe",
   "metadata": {},
   "outputs": [],
   "source": [
    "en_nlp = spacy.load(\"en_core_web_sm\")\n",
    "de_nlp = spacy.load(\"de_core_news_sm\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "ac2a9ce9-772f-4fc1-8859-e818aa01c933",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['What', 'a', 'lovely', 'day', 'it', 'is', 'today', '!']"
      ]
     },
     "execution_count": 18,
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
   "id": "fc43585c-9e4a-4acb-adae-d2bf07a1ad78",
   "metadata": {},
   "source": [
    "### 这个tokenize_example函数用于 对双语（英语和德语）句子进行分词和预处理，通常用于机器翻译任务（如 multi30k 数据集）\n",
    "\n",
    "1.标准化文本处理：统一分词、大小写和长度，便于后续转换为词嵌入或模型输入。\n",
    "\n",
    "2.适配序列模型：添加 sos 和 eos 标记，帮助 Transformer 等模型识别句子边界。\n",
    "\n",
    "3.控制输入维度：通过 max_length 避免过长的序列影响训练效率（如 GPU 内存不足）。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "4fdcb100-e1b9-4a2f-bc5c-46d2833e5c7d",
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
   "id": "7e41df47-8052-49af-b9c0-3bcf314266d7",
   "metadata": {},
   "source": [
    "1.<sos>是Start of Sentence的缩写，表示句子的开始，帮助模型知道何时开始生成输出。\n",
    "\n",
    "2.<eos>\t是End of Sentence的缩写，表示句子的结束，告诉模型何时停止生成。\n",
    "\n",
    "3.dataset.map() 是 Hugging Face datasets 库的方法，用于对数据集中的每个样本应用一个处理函数，此处为tokenize_example"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "80068807-e7cf-4c54-8d64-4c28a323c0ad",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "525ae9881c6945f9b9265f95a4ad0a70",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/29000 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "7e00c911dbcb46f8abfa6c3fd12625e1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/1014 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2d9a17d978ab4405b14a193039fb2a33",
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
   "execution_count": 21,
   "id": "e8b6414f-90c3-47f6-897f-21bfec466dfe",
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
     "execution_count": 21,
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
   "execution_count": 22,
   "id": "0e9379bf-d9e9-4f54-b7e3-5b7b15f64462",
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
   "execution_count": 23,
   "id": "19ea851e-0848-44bf-83a1-f506fccb36ff",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['<unk>', '<pad>', '<sos>', '<eos>', 'a', '.', 'in', 'the', 'on', 'man']"
      ]
     },
     "execution_count": 23,
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
   "execution_count": 24,
   "id": "35414b26-a333-456c-b250-6665fa244813",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['<unk>', '<pad>', '<sos>', '<eos>', '.', 'ein', 'einem', 'in', 'eine', ',']"
      ]
     },
     "execution_count": 24,
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
   "execution_count": 25,
   "id": "8094798c-2cbb-4df9-b768-6f7651c03eb1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7"
      ]
     },
     "execution_count": 25,
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
   "execution_count": 26,
   "id": "5675f2e5-2555-425e-88bc-057d84e1ef6e",
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
   "execution_count": 27,
   "id": "0f747669-4498-43ba-87dd-19a1656026b9",
   "metadata": {},
   "outputs": [],
   "source": [
    "en_vocab.set_default_index(unk_index)\n",
    "de_vocab.set_default_index(unk_index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "f8d893ee-170c-46b2-bc5c-d67c166602f6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[956, 2169, 173, 0, 821]"
      ]
     },
     "execution_count": 28,
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
   "execution_count": 29,
   "id": "312193ee-e3d4-44d0-8e40-748aa27f503d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['i', 'love', 'watching', '<unk>', 'shows']"
      ]
     },
     "execution_count": 29,
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
   "id": "9f05244c-48c0-448f-ad9e-7c7c937c71de",
   "metadata": {},
   "source": [
    "### 这两个单元格的代码共同完成了将分词后的文本转换为数值ID序列的过程。\n",
    "\n",
    "1.第一段代码的作用是将分词后的英文和德文单词列表分别转换为对应的数值ID序列：通过查询词汇表（en_vocab和de_vocab）的lookup_indices方法，将example中的\"en_tokens\"和\"de_tokens\"每个单词替换为词汇表中的整数编号（如\"cat\"→42），最终返回包含\"en_ids\"和\"de_ids\"这两个数值序列的新字典。\n",
    "\n",
    "2.这段代码使用fn_kwargs固定传递词汇表参数，通过dataset.map()将numericalize_example函数批量应用到数据集的所有样本上，为每条数据新增数值ID列（\"en_ids\"和\"de_ids\"），同时保留原始的分词列。例如，将[\"the\",\"cat\"]转换为[5,12]这样的数字序列，而原始文本仍可查看。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "04d8bf9c-6976-422b-adc9-bb1f5c27d682",
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
   "execution_count": 31,
   "id": "bc7e2771-6977-4eb5-bcfe-b2266b0e509c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "721132d973594037bb2699d78ccb1326",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/29000 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4553916b009d4ca4a9b8313b10f12fcf",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/1014 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "31b090d9b8d44393a06e4df67120753e",
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
   "execution_count": 32,
   "id": "7795d525-b2e2-4f5d-bf95-ec76c6d9b2f0",
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
     "execution_count": 32,
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
   "execution_count": 33,
   "id": "965cb615-180c-4608-8b22-57e255e140e0",
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
   "execution_count": 34,
   "id": "1696ebb0-ec4f-40cc-aa33-9c4d1be6a6a2",
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
     "execution_count": 34,
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
   "id": "eede8bbe-7715-4155-818e-f0407d6db4f0",
   "metadata": {},
   "source": [
    "### 这两个函数共同完成了数据批处理和数据加载器创建的工作，是准备训练数据的关键步骤。\n",
    "\n",
    "#### 1、get_collate_fn函数（数据批处理函数工厂），生成一个专门处理序列数据的collate函数。\n",
    "\n",
    "核心功能：接收一个batch的样本(包含不等长的en_ids和de_ids)，使用pad_sequence进行填充对齐(padding_value指定填充值)，返回填充后的整齐batch数据。\n",
    "\n",
    "#### 2、get_data_loader函数（数据加载器创建函数），创建可直接用于训练的数据加载器\n",
    "\n",
    "核心功能：调用get_collate_fn获取定制的批处理函数，创建DataLoader实例并配置关键参数如下：\n",
    "\n",
    "dataset：要加载的数据集\n",
    "\n",
    "batch_size：批大小\n",
    "\n",
    "collate_fn：使用上面创建的定制批处理函数\n",
    "\n",
    "shuffle：是否打乱数据"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "03dbcbd4-4987-49e9-99ad-6ca2aec5138b",
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
   "execution_count": 36,
   "id": "cb4c33f8-64b8-4929-8d85-46419f9429eb",
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
    "    return data_loader\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "e27cf269-131f-42a3-9b0a-8472250d5d29",
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
   "id": "9d7c3c9c-63c5-4bcc-b770-f218c222a83a",
   "metadata": {},
   "source": [
    "### Encoder类实现了一个基于LSTM的序列编码器，主要用于将输入的符号序列编码为隐藏状态表示。\n",
    "\n",
    "1. **输入参数：**\n",
    "\n",
    "+ input_dim：源语言词汇表大小\n",
    "\n",
    "+ embedding_dim：词嵌入维度\n",
    "\n",
    "+ hidden_dim：LSTM隐藏层维度\n",
    "\n",
    "+ n_layers：LSTM层数\n",
    "\n",
    "+ dropout：dropout概率\n",
    "\n",
    "2. **核心组件：**\n",
    "\n",
    "+ nn.Embedding：将离散的单词索引转换为连续的词向量\n",
    "\n",
    "+ nn.LSTM：处理变长序列的双向LSTM\n",
    "\n",
    "+ nn.Dropout：防止过拟合\n",
    "\n",
    "3. **forward处理流程：**\n",
    "\n",
    "+ 输入形状转换：`# src = [src length, batch size]`\n",
    "\n",
    "+ 词嵌入+dropout：`embedded = self.dropout(self.embedding(src))`\n",
    "\n",
    "+ LSTM编码：`outputs, (hidden, cell) = self.rnn(embedded)`\n",
    "\n",
    "其中：\n",
    "\n",
    "outputs：所有时间步的顶层隐藏状态\n",
    "\n",
    "hidden：最后时间步的所有层隐藏状态\n",
    "\n",
    "cell：最后时间步的所有层细胞状态\n",
    "\n",
    "4. **输出说明：**\n",
    "\n",
    "返回元组`(hidden, cell)`：\n",
    "\n",
    "hidden形状：`[n layers, batch size, hidden dim]`\n",
    "\n",
    "cell形状：`[n layers, batch size, hidden dim]`\n",
    "\n",
    "PS：如果是双向LSTM，`n layers`需要乘以2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "d96d6bf9-4711-485b-a86c-e7f3555b727f",
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
   "id": "2e13f119-a2c6-43a7-bf06-4eb5a40cfca8",
   "metadata": {},
   "source": [
    "###  Decoder类的工作流程详解\n",
    "\n",
    "1. **初始化阶段：**\n",
    "\n",
    "- **参数说明：**\n",
    "\n",
    "    + output_dim：目标语言词汇表大小\n",
    "\n",
    "    + embedding_dim：词嵌入维度（需与Encoder一致）\n",
    "\n",
    "    + hidden_dim：LSTM隐藏层维度（需与Encoder一致）\n",
    "\n",
    "    + n_layers：LSTM层数（需与Encoder一致）\n",
    "\n",
    "    + dropout：dropout概率\n",
    "\n",
    "2. **核心组件：**\n",
    "\n",
    "- nn.Embedding：将目标语言的单词索引转换为词向量\n",
    "\n",
    "- nn.LSTM：单步解码的LSTM单元（与Encoder结构对称）\n",
    "\n",
    "- nn.Linear：将隐藏状态映射到词汇表空间（fc_out）\n",
    "\n",
    "- nn.Dropout：正则化层\n",
    "\n",
    "3. **forward流程：**\n",
    "\n",
    "a) 输入预处理：`input = input.unsqueeze(0)  # [batch_size] -> [1, batch_size]`，将当前时间步的输入（单个单词索引）扩展为序列形式。\n",
    "\n",
    "b)词嵌入+dropout：`embedded = self.dropout(self.embedding(input))  # [1, batch_size, embedding_dim]`\n",
    "\n",
    "c) LSTM解码：`output, (hidden, cell) = self.rnn(embedded, (hidden, cell))`，接收来自Encoder的hidden/cell作为初始状态，输出当前时间步的隐藏状态output和更新后的LSTM状态。\n",
    "\n",
    "d) 词汇预测：`prediction = self.fc_out(output.squeeze(0))  # [batch_size, output_dim]`，通过全连接层计算每个单词的预测概率分布。\n",
    "\n",
    "**4.输出说明：**\n",
    "\n",
    "- prediction：当前时间步的词汇概率分布（用于计算loss）\n",
    "\n",
    "- hidden/cell：更新后的LSTM状态（传递给下一个时间步）"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "d01083e6-d8d4-47a0-950c-e37f2d4df0b8",
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
   "id": "b6bdb82a-bd72-4eed-943d-d915778b25ba",
   "metadata": {},
   "source": [
    "### `Seq2Seq`类的详细解释\n",
    "\n",
    "1. **类结构概述**\n",
    "\n",
    "`Seq2Seq`类是一个标准的编码器-解码器架构，包含：\n",
    "\n",
    "- 编码器（encoder）：将源语言序列编码为上下文向量\n",
    "\n",
    "- 解码器（decoder）：基于上下文向量逐步生成目标语言序列\n",
    "\n",
    "- 设备（device）：指定模型运行设备（CPU/GPU）\n",
    "\n",
    "- 一致性检查：确保编码器和解码器的隐藏层维度和层数匹配\n",
    "\n",
    "2. **forward函数流程**\n",
    "\n",
    "**输入：**\n",
    "\n",
    "- `src`：源语言序列，形状为[src长度, batch大小]\n",
    "\n",
    "- `trg`：目标语言序列，形状为[trg长度, batch大小]\n",
    "\n",
    "- `teacher_forcing_ratio`：teacher forcing概率（0.0~1.0）\n",
    "\n",
    "**处理步骤：**\n",
    "\n",
    "a)初始化输出张量：`outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)`，创建一个全零张量用于存储所有时间步的解码器输出。\n",
    "\n",
    "b)编码阶段：`hidden, cell = self.encoder(src)`，编码器处理源语言序列，返回最终的隐藏状态和细胞状态，作为解码器的初始状态。\n",
    "\n",
    "c)解码阶段：\n",
    "\n",
    "+ 初始输入：使用目标序列的第一个token（通常是`<sos>`），`input = trg[0, :]`\n",
    "+ 循环解码：\n",
    "``` python\n",
    "for t in range(1, trg_length):\n",
    "    output, hidden, cell = self.decoder(input, hidden, cell)\n",
    "    outputs[t] = output\n",
    "    # teacher forcing逻辑\n",
    "    teacher_force = random.random() < teacher_forcing_ratio\n",
    "    top1 = output.argmax(1)  # 获取预测最可能的token\n",
    "    input = trg[t] if teacher_force else top1\n",
    " ```\n",
    "- 每个时间步：\n",
    "\n",
    "    + 解码器接收当前输入和LSTM状态，生成输出和新状态\n",
    "\n",
    "    + 存储当前时间步的输出\n",
    "\n",
    "    + 根据teacher_forcing_ratio决定下一个输入是真实token还是预测token\n",
    "\n",
    "**输出：**\n",
    "\n",
    "- outputs：形状为`[trg长度, batch大小, trg词汇表大小]`的张量，包含每个时间步对目标词汇的预测概率分布。\n",
    "\n",
    "3. **Teacher Forcing机制**\n",
    "\n",
    "Teacher Forcing是一种训练技巧，其核心思想是：\n",
    "\n",
    "- 概率性选择输入：\n",
    "\n",
    "    + 以teacher_forcing_ratio的概率使用真实目标序列中的token作为下一个解码器输入。\n",
    "\n",
    "    + 以1 - teacher_forcing_ratio的概率使用解码器自己的预测作为下一个输入。\n",
    "\n",
    "- 作用：\n",
    "\n",
    "    + 使用真实token（teacher forcing）可以加速模型收敛，防止早期训练时错误累积。\n",
    "\n",
    "    + 使用预测token（非teacher forcing）可以让模型学习处理自己的错误，提高鲁棒性。\n",
    " \n",
    "- 典型设置：\n",
    "\n",
    "    + 训练初期：高`teacher_forcing_ratio`（如0.75）\n",
    "\n",
    "    + 训练后期：逐步降低该比例\n",
    "\n",
    "    + 推理阶段：始终使用预测`token（teacher_forcing_ratio=0）`\n",
    " \n",
    "4. 设计要点\n",
    "\n",
    "- 状态传递：编码器的最终状态完整传递给解码器，包含源序列的全部上下文信息。\n",
    "\n",
    "- 序列生成：解码是自回归过程，每个时间步依赖前一步的输出。\n",
    "\n",
    "- 灵活性：通过teacher_forcing_ratio可以灵活控制训练行为。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "b9998669-c5d7-4cae-99b9-ac74a820f73d",
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
   "execution_count": 41,
   "id": "2535b04e-39ad-4952-8567-23ad83f26098",
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
    "\n",
    "# 编码器初始化\n",
    "encoder = Encoder(\n",
    "    input_dim,\n",
    "    encoder_embedding_dim,\n",
    "    hidden_dim,\n",
    "    n_layers,\n",
    "    encoder_dropout,\n",
    ")\n",
    "\n",
    "# 解码器初始化\n",
    "decoder = Decoder(\n",
    "    output_dim,\n",
    "    decoder_embedding_dim,\n",
    "    hidden_dim,\n",
    "    n_layers,\n",
    "    decoder_dropout,\n",
    ")\n",
    "\n",
    "# Seq2Seq模型整合\n",
    "model = Seq2Seq(encoder, decoder, device).to(device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "2d535731-9551-4f9f-981b-644a66fbdd3c",
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
     "execution_count": 42,
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
   "execution_count": 43,
   "id": "d0891ffe-214c-48ac-b391-254c31e3c60b",
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
   "execution_count": 44,
   "id": "2be14455-3b63-45be-abee-4d80140aaa7a",
   "metadata": {},
   "outputs": [],
   "source": [
    "optimizer = optim.Adam(model.parameters())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "61cea329-6297-4285-81b1-5e93e5026e10",
   "metadata": {},
   "outputs": [],
   "source": [
    "criterion = nn.CrossEntropyLoss(ignore_index=pad_index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "f887bf8d-8875-47c4-965d-ef76b4523675",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 定义训练函数\n",
    "def train_fn(\n",
    "    model,           # 要训练的模型\n",
    "    data_loader,     # 数据加载器，用于批量加载训练数据\n",
    "    optimizer,       # 优化器，用于更新模型参数\n",
    "    criterion,       # 损失函数，用于计算模型输出与目标之间的差异\n",
    "    clip,           # 梯度裁剪的阈值，防止梯度爆炸\n",
    "    teacher_forcing_ratio,  # 教师强制比例，控制使用真实标签作为解码器输入的概率\n",
    "    device          # 计算设备（如'cuda'或'cpu'）\n",
    "):\n",
    "    # 将模型设置为训练模式（这会启用dropout和batch normalization等训练特有的层）\n",
    "    model.train()\n",
    "    \n",
    "    # 初始化epoch损失为0\n",
    "    epoch_loss = 0\n",
    "    \n",
    "    # 遍历数据加载器中的每个批次\n",
    "    for i, batch in enumerate(data_loader):\n",
    "        # 从批次中获取源语言（德语）数据并移动到指定设备\n",
    "        src = batch[\"de_ids\"].to(device)\n",
    "        # 从批次中获取目标语言（英语）数据并移动到指定设备\n",
    "        trg = batch[\"en_ids\"].to(device)\n",
    "        # src的形状 = [src长度, 批次大小]\n",
    "        # trg的形状 = [trg长度, 批次大小]\n",
    "        \n",
    "        # 清除优化器中的梯度（防止梯度累积）\n",
    "        optimizer.zero_grad()\n",
    "        \n",
    "        # 前向传播：将源数据和目标数据输入模型\n",
    "        # teacher_forcing_ratio控制是否使用教师强制\n",
    "        output = model(src, trg, teacher_forcing_ratio)\n",
    "        # output的形状 = [trg长度, 批次大小, trg词汇表大小]\n",
    "        \n",
    "        # 获取输出维度（目标词汇表大小）\n",
    "        output_dim = output.shape[-1]\n",
    "        \n",
    "        # 重塑输出：忽略第一个token（通常是<bos>），并将结果展平\n",
    "        output = output[1:].view(-1, output_dim)\n",
    "        # output的形状 = [(trg长度 - 1) * 批次大小, trg词汇表大小]\n",
    "        \n",
    "        # 重塑目标：忽略第一个token（<bos>），并将结果展平\n",
    "        trg = trg[1:].view(-1)\n",
    "        # trg的形状 = [(trg长度 - 1) * 批次大小]\n",
    "        \n",
    "        # 计算损失：模型输出与真实目标之间的差异\n",
    "        loss = criterion(output, trg)\n",
    "        \n",
    "        # 反向传播：计算梯度\n",
    "        loss.backward()\n",
    "        \n",
    "        # 梯度裁剪：防止梯度爆炸\n",
    "        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)\n",
    "        \n",
    "        # 更新模型参数\n",
    "        optimizer.step()\n",
    "        \n",
    "        # 累加当前批次的损失值\n",
    "        epoch_loss += loss.item()\n",
    "    \n",
    "    # 返回整个epoch的平均损失\n",
    "    return epoch_loss / len(data_loader)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "6b7e35b0-1490-48a8-b69d-0e8f341ac692",
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
   "execution_count": 59,
   "id": "d0cd21eb-e01d-407b-a662-772255f22db5",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|███████████████████████████████████████████████████████████████████████████████████| 1/1 [10:27<00:00, 627.03s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\tTrain Loss:   3.625 | Train PPL:  37.507\n",
      "\tValid Loss:   4.155 | Valid PPL:  63.727\n"
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
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "model = model.to(device)  # 将模型移至GPU\n",
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
    "    print(f\"\\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "89f31ab2-9d47-4989-b992-7d7cfb427c75",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<All keys matched successfully>"
      ]
     },
     "execution_count": 60,
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
   "execution_count": 61,
   "id": "60f0f5b0-9e47-4296-901b-9e381b4d021c",
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
   "execution_count": 62,
   "id": "0e18d156-991b-41f5-8608-32baf442079d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "('Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt.',\n",
       " 'A man in an orange hat starring at something.')"
      ]
     },
     "execution_count": 62,
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
   "execution_count": 63,
   "id": "0b3922ee-7eb5-4c0e-9e6e-a0260c7fb768",
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
   "execution_count": 64,
   "id": "492f5447-6431-4225-84aa-055ba51f27eb",
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
       " 'blue',\n",
       " 'shirt',\n",
       " 'is',\n",
       " 'a',\n",
       " 'a',\n",
       " '.',\n",
       " '.',\n",
       " '<eos>']"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "translation"
   ]
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
