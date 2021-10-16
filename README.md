Implementation of the paper: [DORA: Toward Policy Optimization for Task-oriented Dialogue System with Efficient Context](https://arxiv.org/abs/2107.03286).

## Model Architecture

The model consists of a shared encoder, domain state tracker, belief tracker, DB operator, dialogue policy, and response generator.

<figure class="align-center" style="width:600px">
  <img src="images\architecture.png" alt="">
</figure> 

## Requirements

```
python == 3.6
torch == 1.6.0
transformers == 2.9.0
nltk
numpy
```

## Create Data

Unzip `data/MultiWOZ_2.0.zip` and `data/MultiWOZ_2.1.zip`.

**MultiWOZ 2.0**

```
python create_data.py --data_path="data/MultiWOZ_2.0"
```

**MultiWOZ 2.1**

```
python create_data.py
```

## Training

### SL step

```
python train.py
```

* `weight_tying`: use embedding matrix for decoding words
* `attention_projection_layer`:  use additional projection layer for attention

**Configuration of experiments in paper**

```
python train.py --batch_size=8 --lr=3e-5 --dropout=0.2 --gradient_clipping=10 --weight_tying --attention_projection_layer
```

**Multi-GPU**

```
python distributed_train.py --num_gpus=2
```

### RL step

**w/o action rate**

```
python reinforce.py --lr=1e-2 --gradient_clipping=1 --save_path="save/~~~.pt"
```

**Basic action method**

```
python reinforce.py --lr=1e-2 --gradient_clipping=1 --use_action_rate --save_path="save/~~~.pt"
```

**Configuration of experiments in paper**

```
python reinforce.py --batch_size=8 --lr=1e-2 --dropout=0.2 --gradient_clipping=1 --weight_tying --attention_projection_layer --use_action_rate --beta=1e-2 --gamma=0.99 --save_path="save/~~~.pt"
```

**Negative action method**

```
python reinforce.py --lr=1e-2 --gradient_clipping=1 --use_action_rate --negative_action_reward --save_path="save/~~~.pt"
```

**Weighted action method**

```
python reinforce.py --lr=1e-2 --gradient_clipping=1 --use_action_rate --weighted_action_reward --save_path="save/~~~.pt"
```

## Evaluation

```
python test.py --save_path="save/~~~.pt"
```

**Configuration of experiments in paper**

```
python test.py --weight_tying --attention_projection_layer --save_path="save/~~~.pt"
```

**Action control**

```
python test.py --postprocessing --save_path="save/~~~.pt"
```

## Testing

```
python chat.py --save_path="save/~~~.pt"
```

