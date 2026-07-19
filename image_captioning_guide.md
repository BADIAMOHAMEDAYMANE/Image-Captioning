# Image Captioning Project – The Complete Guide

*Written for a student who knows ML and Deep Learning basics (CNNs, RNNs, LSTMs, Transformers) but has never built an image captioning system.*

---

## 1. Project Overview

### What is Image Captioning?

Image Captioning is the task of teaching a computer to **look at a picture and describe it in a full sentence**.

Example:

```
Input:  [photo of a dog running on a beach]
Output: "A brown dog is running on the sand near the water."
```

This sounds simple, because humans do it without effort. But for a machine it is hard. The model must:

1. **See** — understand what objects are in the image (dog, sand, water).
2. **Understand relations** — the dog is *running*, it is *on* the beach.
3. **Speak** — produce a correct, natural sentence, one word at a time.

A useful analogy: imagine you describe a photo to a friend on the phone. Your **eyes** do the seeing, and your **mouth** does the speaking. An image captioning model has the same two parts: an **encoder** (the eyes) and a **decoder** (the mouth).

### Why is it a multimodal problem?

A *modality* is a type of data: images, text, audio, video. Most classic ML problems use one modality only:

- Image classification: image → label (only vision)
- Machine translation: text → text (only language)

Image Captioning uses **two modalities at the same time**: it takes an **image** as input and produces **text** as output. The model must build a "bridge" between the visual world and the language world. This bridge is the heart of the project, and it is why researchers call it a **multimodal** problem. It combines the two biggest fields of deep learning: **Computer Vision** and **Natural Language Processing (NLP)**.

### Real-world applications

| Domain | How captioning helps |
|---|---|
| **Accessibility** | Screen readers describe photos to blind and low-vision users (e.g., Facebook's automatic alt-text). |
| **Autonomous driving** | The car explains a scene in words: "a pedestrian is crossing on the left" — useful for logging and decisions. |
| **Robotics** | A robot describes what it sees so a human can supervise it, or so it can reason about its environment. |
| **Social media** | Automatic alt-text, content indexing, better image search ("find my beach photos"). |
| **Healthcare** | Draft reports for X-rays or scans: "opacity in the lower left lung" — a doctor then reviews them. |
| **Security** | Turn camera video into searchable text logs. |
| **E-commerce** | Generate product descriptions from photos. |

---

## 2. Project Workflow

The full pipeline, from photo to sentence:

```
+-------------+     +----------------+     +------------------+     +----------------+
|   Input     | --> |  Image         | --> |  Feature         | --> |  Text          |
|   Image     |     |  Preprocessing |     |  Extraction      |     |  Generation    |
| (JPEG/PNG)  |     |  (resize, norm)|     |  (CNN or ViT)    |     |  (LSTM/Transf.)|
+-------------+     +----------------+     +------------------+     +--------+-------+
                                                                             |
                                                                             v
                                                                  +--------------------+
                                                                  |  Caption Output    |
                                                                  | "A dog runs on..." |
                                                                  +--------------------+
```

### Step 1 — Image preprocessing

Neural networks need clean, uniform input. So we:

- **Resize** every image to a fixed size (e.g., 224×224 pixels).
- **Convert** to a tensor (a grid of numbers).
- **Normalize** pixel values, usually with ImageNet statistics (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]). We do this because our pretrained encoder was trained with these values — the input must "look the same" as its training data.

### Step 2 — Feature extraction (the encoder)

A pretrained CNN (like ResNet50) or a Vision Transformer reads the image and compresses it into **features**: a compact set of numbers that describe *what* is in the image (shapes, objects, textures), not *raw pixels*. For a 224×224 image, this can be a single vector of 2048 numbers, or a grid of region vectors (more on this in Section 4).

### Step 3 — Text generation (the decoder)

The decoder (LSTM or Transformer) receives the image features and generates the caption **one word at a time**:

```
Features + "<start>"            → "a"
Features + "<start> a"          → "dog"
Features + "<start> a dog"      → "runs"
...
Features + "<start> a dog runs on the beach" → "<end>"
```

Each new word depends on the image *and* on the words already produced. Generation stops when the model outputs the special `<end>` token or reaches a maximum length.

### Step 4 — Caption output

The generated token IDs are converted back to words, special tokens are removed, and we get the final sentence. At inference time we choose words with **greedy search** (always take the most probable word) or **beam search** (keep several candidate sentences alive — see Section 15).

---

## 3. Datasets

Three datasets dominate this field. All of them contain images with **human-written captions** (usually 5 per image, written by different people).

### Flickr8k

- **Images:** ~8,000
- **Captions:** 5 per image (~40,000 total)
- **Content:** everyday scenes — people, dogs, sports, streets.
- **Advantages:** small (~1 GB), trains in hours on a single GPU or even Google Colab. Perfect for testing your pipeline quickly.
- **Limits:** small vocabulary; models overfit easily; captions are simple.

### Flickr30k

- **Images:** ~31,000
- **Captions:** 5 per image (~155,000 total)
- **Advantages:** same style as Flickr8k but ~4× more data → better generalization, richer vocabulary. A good middle step.
- **Limits:** still much smaller than COCO; training takes longer than Flickr8k.

### MS COCO (Common Objects in Context)

- **Images:** ~123,000 (2017 split: 118k train + 5k validation)
- **Captions:** 5 per image (~600,000+ total)
- **Advantages:** the **standard benchmark** — almost every research paper reports COCO scores, so you can compare your model with published results. Large, diverse, complex scenes with many objects.
- **Limits:** ~20+ GB, needs a good GPU and days of training.

### Which one for beginners?

**Start with Flickr8k.** The reason is practical: your first goal is to make the *whole pipeline work* (data loading, training loop, inference, evaluation), and you want fast feedback when you debug. Once everything works on Flickr8k, moving to Flickr30k or COCO is only a config change.

```
Learning path:  Flickr8k  →  Flickr30k  →  MS COCO
                (prototype)   (improve)     (benchmark)
```

**Note on splits:** for Flickr8k/30k, most papers use the **Karpathy split**, a standard train/val/test division. Use it so your results are comparable.

---

## 4. Image Encoder

### Why not raw pixels?

A 224×224 RGB image has 224 × 224 × 3 = **150,528 numbers**. Feeding these directly to a language decoder is a bad idea:

- **Too many inputs** → huge, slow, overfitting model.
- **Pixels are low-level.** Pixel #4,213 being "slightly blue" says nothing about a dog. The decoder needs *concepts* (dog, beach, running), not colors of single points.
- **No spatial understanding.** Shift the dog 10 pixels to the left and all pixel values change, but the meaning is the same. CNNs are built to be robust to this; raw-pixel models are not.

So we use an encoder that transforms pixels into **semantic features**.

### Popular encoders

| Encoder | Feature size | Input size | Notes |
|---|---|---|---|
| **ResNet50** | 2048 | 224×224 | The classic default. Skip connections allow deep training. Great speed/quality balance. |
| **InceptionV3** | 2048 | 299×299 | Multi-scale filters in parallel; used in the famous "Show and Tell" paper. |
| **EfficientNet (B0–B7)** | 1280–2560 | varies | Best accuracy per FLOP; scale it up or down by choosing B0…B7. |
| **ViT (Vision Transformer)** | 768 × 196 patches | 224×224 | Splits image into 16×16 patches, applies self-attention. State of the art with enough data. |

### Feature maps vs global feature vectors

A CNN like ResNet50 processes an image through many layers. Just **before** the final classification layer, it produces a **feature map**:

```
Input image          Feature map              Global vector
(224×224×3)   -->    (7 × 7 × 2048)    -->    (2048)
                     49 regions,              one summary of
                     each described           the whole image
                     by 2048 numbers          (average pooling)
```

- The **feature map** (7×7×2048) is a grid: each of the 49 cells describes one *region* of the image. This is what **attention models** use (Section 6), because they need to look at different regions.
- The **global feature vector** (2048) is one summary of the whole image, made by averaging the grid. Simple CNN+LSTM models use this.

Key intuition: early CNN layers detect edges and textures; deep layers detect object parts and whole objects. We take features from the deep layers because they carry *meaning*.

### Transfer Learning

We never train the encoder from scratch. Instead:

1. Take a CNN **pretrained on ImageNet** (1.28M images, 1000 classes). It already knows how to recognize objects, textures, animals, vehicles…
2. **Remove** its classification head (the last layer).
3. **Freeze** its weights (at least at the start) and use it only to extract features.
4. Optionally, **fine-tune** its last layers later with a small learning rate (Section 15).

Why? Our captioning datasets are small (8k–120k images). Learning good visual features needs millions of images. Transfer learning gives us those features for free — like hiring an experienced photographer instead of teaching a baby to see.

> **Practical tip:** since the frozen encoder always gives the same output for the same image, you can **precompute all features once** and save them to disk. Training then becomes much faster because you skip the CNN in every epoch.

---

## 5. Caption Decoder

The decoder is a **language model conditioned on the image**. It generates the caption word by word. At each step it answers one question: *"given the image and the words so far, what is the next word?"*

```
Step:      1        2        3        4        5         6
Input:  <start>     a       dog     runs      on       the
           |        |        |        |        |         |
           v        v        v        v        v         v
        [ LSTM ]->[ LSTM ]->[ LSTM ]->[ LSTM ]->[ LSTM ]->[ LSTM ]   (hidden state flows →)
           |        |        |        |        |         |
           v        v        v        v        v         v
Output:   "a"     "dog"    "runs"    "on"     "the"    "beach"
```

The image features enter the decoder either as the **initial hidden state**, or as a **first "word"**, or through **attention at every step** (Section 6).

### Vanilla RNN vs LSTM vs GRU

| Property | Vanilla RNN | LSTM | GRU |
|---|---|---|---|
| Gates | none | 3 (forget, input, output) + cell state | 2 (reset, update) |
| Long-term memory | poor | good | good |
| Vanishing gradients | severe | largely solved | largely solved |
| Parameters / speed | fewest / fastest | most / slowest | middle |
| Caption quality | weak | strong (classic choice) | close to LSTM |

### Why does LSTM beat a simple RNN?

A vanilla RNN updates its memory like this: `h_t = tanh(W·[h_{t-1}, x_t])`. Every step, the *entire* memory is rewritten and squashed through `tanh`. Two problems follow:

1. **Vanishing gradients.** During backpropagation through time, gradients are multiplied by small numbers at every step. After 10–15 steps they become almost zero, so the network cannot learn long-distance dependencies. In the caption *"A man **who is wearing a red jacket and blue jeans** is skiing"*, an RNN forgets the man by the time it must say "is skiing".
2. **No selective memory.** The RNN cannot choose what to keep and what to throw away.

The **LSTM** fixes this with a **cell state** — a "conveyor belt" of memory that flows through time almost unchanged — controlled by three gates:

- **Forget gate:** what old information should we erase?
- **Input gate:** what new information should we store?
- **Output gate:** what part of the memory should we use right now?

Because information on the conveyor belt is modified by *addition* (not repeated multiplication), gradients can travel far back without vanishing. Analogy: an RNN is a person who rewrites their whole notebook every minute; an LSTM is a person with a notebook, an eraser, and a highlighter — they decide what to erase, what to add, and what to read.

**GRU** is a lighter LSTM (2 gates, no separate cell state). It trains faster and performs almost the same; a fine choice when compute is limited.

---

## 6. Attention Mechanism

### The problem: one vector is a bottleneck

In the basic CNN+LSTM model, the *entire image* is squeezed into a single 2048-number vector, and the decoder must generate the *whole caption* from it. For a complex scene ("a woman throws a frisbee to a dog while children play behind them"), one vector is simply not enough — details get lost. Also, it is unnatural: when the model says "frisbee", why should it think about the children?

Humans don't work this way. When you describe a photo, your **eyes jump around**: you look at the dog when you say "dog", at the beach when you say "beach".

### The idea: look at the right region for each word

**Attention** lets the decoder do exactly this. Instead of one global vector, the encoder provides the **feature map**: 49 region vectors (the 7×7 grid). At *every generation step*, the decoder computes a **weight for each region** — "how relevant is this region for the word I am about to produce?" — and builds a weighted sum called the **context vector**.

```
Generating: "A dog runs on the beach"

Word "dog":                        Word "beach":
 region weights (7×7):              region weights (7×7):
 . . . . . . .                      # # # . . # #
 . . # # . . .    ← high weight     # # . . . # #
 . . # # # . .      on the dog      # . . . . . #
 . . . # . . .                      . . . . . . .
 . . . . . . .                      . . . . . . .
 (focus: center,                    (focus: background,
  where the dog is)                  sand and water)
```

If you plot these weights as a heatmap over the image, you can literally *see where the model is looking* — one of the most satisfying visualizations in deep learning (the "Show, Attend and Tell" paper is full of them).

### Bahdanau (additive) attention — the math, gently

Introduced for machine translation (Bahdanau et al., 2015) and adapted to captioning in "Show, Attend and Tell" (Xu et al., 2015). At decoding step *t*:

```
inputs:  a_1 ... a_49   (region vectors from the CNN, each 2048-d)
         h_{t-1}         (decoder hidden state = "what I've said so far")

1. score each region:      e_i = v^T · tanh(W_a·a_i + W_h·h_{t-1})
2. normalize to weights:   α_i = softmax(e_i)          (all α_i ≥ 0, sum = 1)
3. context vector:         c_t = Σ α_i · a_i           (weighted sum)
4. next word:              LSTM input = [embedding(prev word), c_t]
```

Intuition: the hidden state asks a question ("I need a location word next"), each region gives an answer, and the softmax decides which answers to trust. The whole thing is differentiable, so it trains end-to-end with normal backpropagation.

### Soft vs Hard attention

| | Soft attention | Hard attention |
|---|---|---|
| How it works | weighted **average** of all regions | **sample one** region (stochastic) |
| Differentiable? | yes → normal backprop | no → needs REINFORCE (policy gradient) |
| Training | easy, stable | tricky, high variance |
| Focus | smooth, slightly blurry | sharp, discrete |
| Used in practice | **almost always** | rare |

**Recommendation:** implement soft attention. Hard attention is mainly of historical/research interest.

Attention also solved a deeper problem: it removed the fixed-size bottleneck between encoder and decoder. This idea became so powerful that researchers asked: *what if attention is all you need?* — which brings us to Transformers.

---

## 7. Transformers

### Why Transformers replaced RNN/LSTM

LSTMs have two structural problems:

1. **Sequential processing.** Word 10 cannot be computed before word 9. Training cannot be parallelized along the sequence → slow on GPUs, which love parallel work.
2. **Long-range paths.** Information from word 1 must pass through 9 steps to influence word 10. Even with gates, some signal is lost.

The Transformer ("Attention Is All You Need", 2017) removes recurrence completely. Every word connects **directly** to every other word through **self-attention** — path length 1, fully parallel training.

### Self-Attention

Each token creates three vectors from its embedding:

- **Query (Q):** "what am I looking for?"
- **Key (K):** "what do I contain?" (like a label)
- **Value (V):** "what information do I give if selected?"

```
Attention(Q, K, V) = softmax( Q·K^T / √d_k ) · V
```

Each token compares its Query with everyone's Keys (dot product), turns the scores into weights (softmax), and takes a weighted sum of Values. Analogy: a library search — your Query is the search text, Keys are book titles, Values are the books' contents. The `√d_k` division just keeps the scores in a healthy range so softmax doesn't saturate.

In the sentence *"The dog chased the ball because **it** was fast"*, self-attention lets "it" directly attend to "ball" and resolve the reference in one step.

### Multi-Head Attention

Instead of one attention, we run **h parallel attentions** (e.g., 8 heads), each with its own smaller Q/K/V projections, then concatenate the results. Each head learns to look for something different: one head may track grammar (subject↔verb), another may track object relations, another positions. Like 8 specialists reading the same sentence, each with their own focus.

### Positional Encoding

Self-attention is a *set* operation — it has no idea of word order. "Dog bites man" and "man bites dog" would look identical! So we **add a position signal** to each embedding. The classic choice is sine/cosine waves of different frequencies:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Each position gets a unique "fingerprint," and the pattern generalizes to positions longer than those seen in training. Many modern models simply use **learned** position embeddings instead — both work.

### Transformer Decoder (for captioning)

Each decoder layer has three sub-blocks:

1. **Masked self-attention** over the words generated so far. The *mask* stops position *t* from seeing positions > *t* (no cheating by looking at future words).
2. **Cross-attention:** words are Queries, **image features are Keys and Values**. This is where the text "looks at" the image — the Transformer's version of Bahdanau attention.
3. **Feed-forward network** applied to each position, with residual connections and layer normalization around everything.

### Vision Transformer (ViT)

ViT applies the Transformer to images:

```
Image 224×224 → cut into 196 patches of 16×16 → flatten each patch
→ linear projection to 768-d ("patch embeddings") → add positional encoding
→ standard Transformer encoder → 196 output vectors (one per patch)
```

A patch is treated like a "word" of the image. Unlike a CNN (which only looks at local neighborhoods and grows its view layer by layer), every patch can attend to every other patch **from layer 1** — global context immediately. ViT needs large pretraining data (ImageNet-21k or bigger) to beat CNNs, which is why we always use pretrained ViT weights.

### CNN+LSTM vs Transformer-based captioning

| Aspect | CNN + LSTM (+attention) | ViT + Transformer decoder |
|---|---|---|
| Image encoder | CNN grid features (7×7×2048) | 196 patch embeddings |
| Text generator | LSTM, one word per step, sequential | Transformer decoder, all positions in parallel (training) |
| Image–text link | attention at each LSTM step | cross-attention in every layer |
| Training speed | slow (sequential) | fast (parallel) — but inference is still word-by-word for both |
| Long captions | memory fades | direct connections everywhere |
| Data appetite | works OK on small data | needs more data / pretrained weights |
| Caption quality | good | state of the art |

---

## 8. Project Evolution

A short history — each step fixes a weakness of the previous one:

```
2014: CNN + RNN              "Show and Tell" idea. Image vector → vanilla RNN.
  |                          Weakness: RNN forgets quickly.
  v
2015: CNN + LSTM             Same, with LSTM. Longer, more correct captions.
  |                          Weakness: whole image squeezed into ONE vector.
  v
2015: CNN + LSTM + Attention "Show, Attend and Tell". Decoder looks at image
  |                          regions per word. Weakness: LSTM is slow, sequential.
  v
2018-20: CNN + Transformer   Transformer decoder over CNN region features.
  |                          Faster training, better long captions.
  v
2020-21: ViT + Transformer   Pure attention, end to end. Patches in, words out.
  |                          Weakness: still trained per-task on one dataset.
  v
2022+: Large pretrained multimodal models (BLIP, BLIP-2, Flamingo, GPT-4V)
```

### The modern era (high-level)

- **BLIP (2022).** Pretrained on ~129M noisy web image–text pairs. Key trick: **bootstrapping** — a captioner model generates cleaner captions for web images and a filter model removes bad pairs, so the model improves its own training data. One architecture handles captioning, retrieval, and VQA.
- **BLIP-2 (2023).** Insight: powerful frozen vision models and frozen LLMs already exist — don't retrain them. BLIP-2 **freezes both** and trains only a small bridge network, the **Q-Former**, that translates visual features into "tokens" the LLM understands. Result: near state-of-the-art captioning at a tiny fraction of the training cost, and the captions inherit the LLM's fluency.
- **Flamingo (DeepMind, 2022).** A visual language model that handles **interleaved** sequences of images and text (like a chat with photos). Uses a frozen LLM (Chinchilla) plus inserted **gated cross-attention** layers and a Perceiver Resampler. Famous for **few-shot learning**: show it 2–3 example image–caption pairs in the prompt and it adapts to a new style without any training.
- **GPT-4 Vision and modern multimodal LLMs (2023+).** Captioning is no longer a separate system — it is one of many abilities of a general multimodal model. The vision encoder's output is projected into the LLM's token space, so the image becomes "words" the LLM can reason about. You can ask for a caption, a joke about the image, or an explanation of a diagram.

**The big trend:** from small task-specific models trained on one dataset → to giant general models pretrained on web-scale data, where captioning is just one skill. Your project rebuilds steps 1–5 of this history, which is exactly how you understand what the big models do inside.

---

## 9. Data Preparation

Text must become numbers before a neural network can use it. The steps:

### 9.1 Caption cleaning

Human captions are messy. We clean them:

```
Original:  "A Man, riding his BIKE!! (fast)"
Cleaned:   "a man riding his bike fast"
```

- Remove punctuation and special characters.
- Remove or normalize numbers (project choice).
- Remove extra spaces.
- Optionally drop single-character noise tokens.

### 9.2 Lowercase conversion

"Dog", "dog" and "DOG" are the same word. Without lowercasing, the model would learn three separate embeddings for one concept — wasted capacity and a bigger vocabulary.

### 9.3 Tokenization

Split each caption into tokens (for this project, simple word-level splitting is enough):

```
"a man riding his bike" → ["a", "man", "riding", "his", "bike"]
```

(Modern large models use subword tokenizers like BPE, which handle rare words better — a nice extension, not needed for a first version.)

### 9.4 Vocabulary creation

Build a dictionary mapping each word to an integer ID. Two rules:

- **Frequency threshold:** keep only words that appear ≥ 5 times (typical). Rare words can't be learned well anyway.
- **Special tokens:** reserve IDs for:

| Token | ID (example) | Purpose |
|---|---|---|
| `<pad>` | 0 | filling short sequences |
| `<start>` | 1 | "begin generating" |
| `<end>` | 2 | "the caption is finished" |
| `<unk>` | 3 | any word not in the vocabulary |

Flickr8k gives a vocabulary of roughly 3,000 words (with threshold 5); COCO ~10,000.

### 9.5 Start and end tokens

Every caption is wrapped:

```
<start> a man riding his bike <end>
```

- `<start>` is the first input to the decoder — without it, there is nothing to feed at step 1.
- `<end>` teaches the model **when to stop**. At inference, generation stops when `<end>` is produced. Forget this token and your model will produce endless sentences.

### 9.6 Padding and maximum sequence length

GPUs process batches, and every sequence in a batch (as a tensor) must have the same length. So:

1. Choose a **maximum length** (e.g., cover ~95% of captions; on Flickr8k something like 35 is typical). Longer captions are truncated.
2. **Pad** shorter captions with `<pad>`:

```
<start> a dog runs <end> <pad> <pad> <pad> ...   (up to max_len)
```

Two important details:

- The loss must **ignore pad positions** (`ignore_index=0` in PyTorch's CrossEntropyLoss). Otherwise the model learns to predict `<pad>` everywhere — a classic bug.
- Final numeric form:

```
"<start> a dog runs <end>"  →  [1, 4, 56, 213, 2, 0, 0, 0]
```

---

## 10. Training

### 10.1 Teacher Forcing

During training, we do **not** feed the model's own (possibly wrong) predictions back as input. We feed the **ground-truth** previous words:

```
Without teacher forcing:            With teacher forcing:
step 1: predicts "the" (wrong)      step 1: predicts "the" (wrong, penalized)
step 2: input = "the"  ← its error  step 2: input = "a"   ← the true word
        errors snowball...                  learning stays on track
```

Analogy: a driving instructor who corrects your steering immediately, instead of letting you drive into a field and learn nothing.

Benefits: stable, fast, and (for Transformers) allows computing **all time steps in parallel** — feed the whole true caption at once with a causal mask.

Cost: **exposure bias** — at inference the model must consume its *own* outputs, a situation it never saw in training. Scheduled sampling (Section 15) is one remedy.

### 10.2 Batch training

Typical batch sizes: 32–128 caption–image pairs. One training example = (image features, caption). Note that each image appears ~5 times per epoch, once per caption. Shuffle every epoch.

### 10.3 Loss function — Cross Entropy

At each position the model outputs a probability distribution over the vocabulary. Cross-entropy compares it with the true word:

```
loss_t = -log P(true_word_t)
```

If the model gives the true word probability 0.9 → loss ≈ 0.105 (good). Probability 0.01 → loss ≈ 4.6 (bad). The total loss is the average over all non-pad positions. This is exactly classification, repeated at every time step, with "classes" = vocabulary words.

A related, more interpretable metric is **perplexity** = exp(loss): roughly "between how many words is the model hesitating?"

### 10.4 Optimizers

- **Adam / AdamW** is the standard choice (adaptive learning rates per parameter, robust defaults). Start with `lr = 3e-4` for the decoder.
- If you fine-tune the encoder, give it a **much smaller** learning rate (e.g., `1e-5`) — its pretrained weights are already good; big steps would destroy them.

### 10.5 Learning rate scheduling

A fixed learning rate is rarely optimal: big steps help early, small steps help late.

- **ReduceLROnPlateau:** cut LR (e.g., ×0.5) when validation loss stops improving. Simple and effective.
- **Warmup + decay** (standard for Transformers): increase LR linearly for the first few thousand steps, then decay. Transformers are unstable at the start; warmup prevents early divergence.
- **Cosine annealing:** smooth decrease following a cosine curve.

### 10.6 Early stopping

Track **validation loss (or BLEU-4)** after each epoch. If it does not improve for *patience* epochs (e.g., 5), stop and keep the **best checkpoint**, not the last one. This prevents overfitting and saves compute. Typical training: 15–30 epochs on Flickr8k.

```
loss
  |  \ train loss keeps falling
  |   \_______
  |    \      ‾‾——————___
  |     \  val loss
  |      \_____   ← best checkpoint here: SAVE THIS
  |            ‾‾/‾‾  ← val loss rises: overfitting, STOP
  +------------------------------ epochs
```

---

## 11. Evaluation

How do we score a generated caption? A caption can be correct in many different ways ("a dog runs on the beach" ≈ "a brown dog is running near the sea"), so we compare it against **all 5 reference captions** and use several metrics.

### BLEU (Bilingual Evaluation Understudy)

Measures **n-gram precision**: what fraction of the generated n-grams appear in the references? BLEU-1 uses single words, BLEU-4 uses 4-word sequences. Includes a *brevity penalty* so the model can't cheat with very short captions.

- **Range:** 0–1 (often shown ×100). A CNN+LSTM+attention model on Flickr8k typically reaches BLEU-4 ≈ 0.20–0.25; strong COCO models ≈ 0.35–0.40.
- **Good for:** quick, standard comparison; every paper reports it.
- **Weakness:** only exact word matches — "sea" vs "ocean" gets zero credit; correlates only moderately with human judgment.

### METEOR

Improves on BLEU: matches also **stems** (run/running) and **synonyms** (via WordNet), and uses both precision **and recall**, plus a word-order penalty.

- **Good for:** fairer credit for meaning; better correlation with humans than BLEU.
- **Weakness:** needs linguistic resources; mostly English-centric; slower.

### ROUGE (ROUGE-L for captioning)

**Recall-oriented**: did the caption cover the content of the references? ROUGE-L uses the **longest common subsequence** between candidate and reference.

- **Good for:** checking the caption doesn't *miss* important content. Comes from text summarization.
- **Weakness:** less discriminative for captioning; usually reported as a secondary metric.

### CIDEr (Consensus-based Image Description Evaluation)

**Designed specifically for image captioning.** Uses **TF-IDF weighted** n-grams: words that are informative for *this* image (like "frisbee") count much more than generic words (like "a", "is"). Measures consensus with all references.

- **Range:** 0 to ~10 in theory; good COCO models score 1.1–1.4.
- **Good for:** the **primary captioning metric** in most modern papers — best n-gram correlation with human judgment. Rewards *specific, informative* captions.
- **Weakness:** can be gamed by repeating high-TF-IDF words; needs multiple references to work well.

### SPICE

Ignores exact words. Parses candidate and references into **scene graphs** — objects, attributes, relations:

```
"a young girl is eating a red apple"
→ objects:    girl, apple
→ attributes: girl:young, apple:red
→ relations:  eat(girl, apple)
```

then computes an F-score between the graphs.

- **Good for:** measuring **semantic correctness** — did the model understand *what is happening*, regardless of wording? Highest correlation with human semantic judgment.
- **Weakness:** ignores fluency completely (a broken ungrammatical caption with the right facts scores high); depends on parser quality; slow.

### Which metric when?

| Situation | Use |
|---|---|
| Quick sanity check during development | BLEU-4 |
| Comparing with published papers | BLEU-4 + **CIDEr** (+ METEOR, SPICE) |
| Caring about meaning, not wording | SPICE, METEOR |
| Final report | all of them, via the `pycocoevalcap` package |

> **Always also look at actual sample captions.** Metrics summarize; your eyes catch failure modes (repetition, hallucinated objects) that numbers hide.

---

## 12. Project Architecture (ASCII diagrams)

### Architecture A — CNN → LSTM ("Show and Tell")

```
        +-----------------+
Image → |  ResNet50       | → global vector (2048) → Linear → (embed_dim)
        |  (frozen)       |                              |
        +-----------------+                              v  (init hidden state)
                                                  +-------------+
<start> → Embedding ────────────────────────────→ |             | → "a"
"a"     → Embedding ────────────────────────────→ |    LSTM     | → "dog"
"dog"   → Embedding ────────────────────────────→ |             | → "runs"
  ...                                             +-------------+    ...
                                                        |
                                                 Linear → softmax over vocab
```

One image summary; the LSTM must remember it for the whole caption.

### Architecture B — CNN → Attention → LSTM ("Show, Attend and Tell")

```
        +-----------------+
Image → |  ResNet50       | → feature map: 49 region vectors (7×7×2048)
        |  (frozen)       |        a_1 ... a_49
        +-----------------+          |
                                     |   (all 49, at EVERY step)
                                     v
                            +------------------+
        h_{t-1} ──────────→ |    ATTENTION     |
        (decoder state)     |  α_i = softmax(  |
                            |   score(a_i,h))  |
                            +--------+---------+
                                     |  context c_t = Σ α_i a_i
                                     v
   prev word → Embedding →  +------------------+
                     concat |      LSTM        | → h_t → Linear → next word
                            +------------------+
                            (repeat for every word)
```

New context vector at every step: "dog" ← dog regions, "beach" ← background regions.

### Architecture C — ViT → Transformer Decoder

```
        +---------------------------------------------+
Image → | ViT: 196 patches → patch embeddings + pos.  | → memory: 196 vectors
        |      → self-attention encoder × N layers    |
        +---------------------------------------------+
                                                     |
                                                     |  (Keys, Values)
                                                     v
 <start> a dog runs ...  (all positions at once, training)
        |
        v
 Token Embedding + Positional Encoding
        |
        v
 +--------------------------------------+
 | Transformer Decoder layer × N:       |
 |  1. Masked self-attention (words)    |
 |  2. Cross-attention (words → image) ←+── memory
 |  3. Feed-forward + residuals + LN    |
 +--------------------------------------+
        |
        v
 Linear → softmax over vocabulary  → "a dog runs on the beach <end>"
```

---

## 13. Implementation Roadmap (Python + PyTorch)

### Step 0 — Project structure

```
image-captioning/
├── data/
│   ├── raw/                  # Flickr8k images + captions.txt
│   ├── processed/            # precomputed features, encoded captions
│   └── vocab.json
├── src/
│   ├── config.py             # all hyperparameters in ONE place
│   ├── dataset.py            # Dataset + DataLoader (+ collate_fn)
│   ├── vocabulary.py         # build vocab, encode/decode
│   ├── models/
│   │   ├── encoder.py        # ResNet/ViT wrappers
│   │   ├── decoder_lstm.py   # LSTM decoder (+ attention)
│   │   └── transformer.py    # Transformer decoder
│   ├── train.py              # training loop, checkpoints
│   ├── evaluate.py           # BLEU/METEOR/CIDEr on test set
│   └── inference.py          # caption a new image (greedy + beam)
├── notebooks/                # exploration, visualization
├── checkpoints/
├── outputs/                  # curves, attention maps, samples
├── app.py                    # deployment (Section 16)
├── requirements.txt
└── README.md
```

### Step 1 — Dataset preparation (Week 1)

1. Download Flickr8k; load `captions.txt` into a dict `{image_id: [5 captions]}`.
2. Clean captions, build vocabulary (threshold ≥ 5), save `vocab.json`.
3. Split train/val/test with the **Karpathy split**.
4. **Precompute image features** with frozen ResNet50 and save as `.pt` files (both the 2048 global vector and the 7×7×2048 map).
5. Write a `CaptionDataset(torch.utils.data.Dataset)` returning `(features, caption_ids, length)` and a `collate_fn` that pads batches.

**Checkpoint test:** print one batch; decode a caption back to words; check `<start>`/`<end>`/`<pad>` are correct.

### Step 2 — Baseline model: CNN + LSTM (Week 2)

```python
class DecoderLSTM(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.init_h = nn.Linear(2048, hidden_dim)   # image → initial state
        self.init_c = nn.Linear(2048, hidden_dim)
        self.lstm   = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc     = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):          # teacher forcing
        h0 = self.init_h(features).unsqueeze(0)
        c0 = self.init_c(features).unsqueeze(0)
        emb = self.embed(captions[:, :-1])           # input: all but <end>
        out, _ = self.lstm(emb, (h0, c0))
        return self.fc(out)                          # target: all but <start>
```

### Step 3 — Training loop (Week 2–3)

```python
criterion = nn.CrossEntropyLoss(ignore_index=0)     # ignore <pad>!
optimizer = torch.optim.AdamW(decoder.parameters(), lr=3e-4)

for epoch in range(cfg.epochs):
    for features, captions, _ in train_loader:
        logits = decoder(features, captions)
        loss = criterion(logits.reshape(-1, vocab_size),
                         captions[:, 1:].reshape(-1))
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)  # stability
        optimizer.step()
    validate(); save_if_best(); early_stop_check()
```

Log train/val loss every epoch (CSV or TensorBoard) — you need these curves for the report.

### Step 4 — Inference (Week 3)

Greedy loop: start with `<start>`, feed the model, take argmax, append, repeat until `<end>` or max_len. Then implement **beam search** (k = 3–5). Test on your own photos — this is the fun part.

### Step 5 — Attention model (Week 3–4)

Add the Bahdanau attention module (Section 6) on top of the 7×7×2048 features; the LSTM input becomes `[word_embedding ; context_vector]`. Save attention weights during inference and plot them as heatmaps over the image.

### Step 6 — Transformer model (Week 4–5)

Use `nn.TransformerDecoder` (PyTorch has it built in) over patch/region features. Don't forget the **causal mask** and positional encodings. Optionally swap ResNet features for ViT (`timm` library: `vit_base_patch16_224`).

### Step 7 — Evaluation (Week 5)

Generate captions for the whole test set with beam search; compute BLEU-1..4, METEOR, ROUGE-L, CIDEr, SPICE with `pycocoevalcap`. Build the comparison table (Section 14) for all your models.

### Step 8 — Error analysis, docs, deployment (Week 6)

Collect best/worst captions, categorize failures, write the README, deploy a demo (Section 16).

---

## 14. Comparison Study

| Criterion | CNN + RNN | CNN + LSTM | CNN + LSTM + Attention | ViT + Transformer |
|---|---|---|---|---|
| **Advantages** | simplest; fewest params; fast | remembers long context; stable training | looks at regions per word; interpretable heatmaps; big quality jump | best quality; parallel training; scales with data; direct long-range links |
| **Disadvantages** | vanishing gradients; short bad captions | one-vector bottleneck; sequential | still sequential; more complex code | data-hungry; needs pretrained weights; most complex |
| **Complexity** (decoder, per layer) | O(T·d²), sequential | O(T·d²), sequential | O(T·d²) + O(T·R·d) attention (R=49 regions) | O(T²·d) self-attn + O(T·P·d) cross-attn (P=196 patches), parallel over T |
| **Memory (training)** | lowest | low (~10–20M params) | medium (+attention states, ~20–30M) | highest (~50–150M+; attention matrices) |
| **Training time** (Flickr8k, 1 GPU) | fastest per epoch, but poor results | baseline (~hours) | ~1.5–2× LSTM | fastest *per epoch* thanks to parallelism, but more epochs/params; overall similar or less |
| **Caption quality** (BLEU-4, Flickr8k, typical) | ~0.12–0.15 | ~0.17–0.21 | ~0.21–0.25 | ~0.25–0.30+ (with pretrained ViT) |
| **Interpretability** | none | none | **attention heatmaps** | attention maps (harder to read, many heads/layers) |

Notes:

- The T² term in Transformers is harmless for captions (T ≤ 35) — it matters for long documents, not here.
- The biggest single quality jump historically was **adding attention**; the second was **large-scale pretraining**, not architecture alone.
- For a university project, implementing all four and producing this table *from your own experiments* is the core scientific contribution.

---

## 15. Best Practices

### Data augmentation

Apply random transforms to training images so the model sees "new" data every epoch: random crop, horizontal flip, small color jitter. **Careful:** the transform must not contradict the caption — a horizontal flip breaks captions containing "left"/"right", and a strong crop may remove the object being described. Keep augmentation mild. (Note: augmentation requires computing CNN features on the fly, so it conflicts with the precompute-features trick — use it only when fine-tuning the encoder.)

### Transfer learning & fine-tuning

The standard two-phase recipe:

1. **Phase 1:** encoder frozen, train only the decoder until validation plateaus.
2. **Phase 2:** unfreeze the encoder's **last block(s)** and continue with a tiny LR (1e-5). Fine-tuning too early or too fast destroys the pretrained features ("catastrophic forgetting").

Typical gain: +1–2 BLEU points from phase 2.

### Greedy Search vs Beam Search

- **Greedy:** at each step take the single most probable word. Fast, but a locally best word can lead to a globally bad sentence — and there is no way back.
- **Beam search (k=3–5):** keep the k best partial sentences at each step; expand all of them; keep the k best again; finally return the complete sentence with the best total (length-normalized) log-probability.

```
Greedy:  keeps 1 path      "a" → "man" → "is" → ...
Beam-3:  keeps 3 paths     "a man is ..."    (score -2.1)
                           "a person on ..." (score -2.3)
                           "two men are ..." (score -2.6)
```

Beam search usually adds +2–4 BLEU. Very large beams (k>10) can paradoxically give *worse*, more generic captions — k=3–5 is the sweet spot. Use **length normalization** (divide score by length), or beams will prefer short captions.

### Scheduled Sampling

A remedy for exposure bias (Section 10.1): during training, with probability *p*, feed the model's **own prediction** instead of the true previous word. Start with p=0 (pure teacher forcing) and increase it gradually across epochs. The model slowly learns to survive its own mistakes. Use a gentle schedule — too aggressive and training destabilizes.

### Mixed Precision Training (AMP)

Compute mostly in float16 instead of float32:

```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    logits = model(features, captions)
    loss = criterion(...)
scaler.scale(loss).backward()
scaler.step(optimizer); scaler.update()
```

Result: ~2× faster training, ~half the GPU memory, essentially no quality loss. The `GradScaler` prevents small gradients from underflowing to zero. Almost free performance — use it.

### Other habits that pay off

- **Gradient clipping** (max norm 5.0) — prevents exploding gradients in RNNs.
- **Config file** for all hyperparameters + fixed random seeds → reproducible experiments.
- **Overfit one batch first:** before full training, check your model can reach ~zero loss on a single batch. If it can't, there is a bug.

---

## 16. Deployment

Turn the model into something people can use. Preparation for all options: save the best checkpoint + `vocab.json`, and write one function `generate_caption(image) -> str` that wraps preprocessing → encoder → beam search → decoding.

### Option A — FastAPI (or Flask): a REST API

Best when the model must serve **other programs** (a website, a mobile app).

```python
from fastapi import FastAPI, UploadFile
from PIL import Image
import io

app = FastAPI()
model = load_model("checkpoints/best.pt")     # load ONCE at startup

@app.post("/caption")
async def caption(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    return {"caption": generate_caption(model, image)}
```

Run with `uvicorn app:app`. FastAPI gives you automatic docs at `/docs`, async support, and input validation — generally preferred over Flask for new projects. For production: Docker container, and `model.eval()` + `torch.no_grad()` in inference.

### Option B — Streamlit: an instant web app

Best for **demos and presentations** — a friendly UI in ~20 lines:

```python
import streamlit as st
from PIL import Image

st.title("🖼️ Image Captioning Demo")
model = st.cache_resource(load_model)("checkpoints/best.pt")

file = st.file_uploader("Upload an image", type=["jpg", "png"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img)
    with st.spinner("Thinking..."):
        st.success(generate_caption(model, img))
```

Run with `streamlit run app.py`. Add a beam-size slider and an attention-heatmap toggle to impress the jury.

### Option C — Hugging Face Spaces: free public hosting

Best for **sharing** — anyone on the internet can try your model.

1. Create a Space at huggingface.co/spaces (choose Gradio or Streamlit SDK).
2. Push `app.py`, `requirements.txt`, and your checkpoint (use Git LFS for files >10 MB).
3. The Space builds and serves automatically at `huggingface.co/spaces/<you>/<name>`.

Free CPU tier is enough: one caption takes 1–2 seconds on CPU. A public demo link in your README/CV is worth a lot.

---

## 17. Final Deliverables

A complete university project should contain:

1. **Source code** — the structure from Section 13, clean, commented, with `requirements.txt` and fixed seeds. All four models selectable from the config.
2. **Documentation** — how to install, download data, train, evaluate, and run the demo. Docstrings on every module.
3. **Model architecture diagrams** — the three diagrams of Section 12 (redrawn nicely with draw.io or similar).
4. **Training curves** — train/val loss per epoch for every model, plus a val-BLEU curve. These prove your training was healthy and show where early stopping fired.
5. **Evaluation metrics** — the full table: BLEU-1..4, METEOR, ROUGE-L, CIDEr, SPICE × 4 models, on the same test split.
6. **Sample predictions** — 10–15 test images with: reference caption, output of each model, and attention heatmaps. Include both successes and failures.
7. **Error analysis** — categorize failures: object hallucination (mentions things not present), wrong counting ("two dogs" for one), wrong colors, generic captions ("a man is standing"), repetition loops. Give the frequency of each category and hypothesize causes.
8. **README** — project banner + demo GIF, motivation, results table, quickstart commands, link to the live demo.
9. **Presentation slides** — 12–15 slides: problem → pipeline → the 4 architectures → results table → attention visualizations → live demo → lessons learned. The attention heatmaps and the live demo are your strongest moments.

---

## Common Implementation Mistakes (read this twice)

1. **Not ignoring `<pad>` in the loss** → the model predicts padding forever. Use `CrossEntropyLoss(ignore_index=0)`.
2. **Off-by-one in teacher forcing** → inputs must be `caption[:-1]`, targets `caption[1:]`. Get this wrong and the model "predicts" the word it was just given (train loss suspiciously near zero, garbage at inference).
3. **Forgetting `<end>`** → captions never stop.
4. **Wrong normalization** → using different mean/std than the pretrained encoder expects silently degrades features.
5. **Building vocabulary on train+test** → data leakage; use training captions only.
6. **Forgetting the causal mask** in the Transformer decoder → the model sees future words during training, then fails completely at inference.
7. **Forgetting `model.eval()` + `torch.no_grad()`** at inference → dropout stays active, results are random and memory explodes.
8. **Fine-tuning the encoder from epoch 1 with a big LR** → destroys pretrained features.
9. **Evaluating with only one reference caption** → unfairly low scores; use all 5.
10. **Trusting metrics without reading captions** → always inspect real outputs.

---

## Extending the Project to Research Level

Once the four baselines work, natural next steps:

- **Subword tokenization (BPE)** — removes the `<unk>` problem, handles rare words.
- **Bottom-up attention** — use an object detector (Faster R-CNN) to attend over *object proposals* instead of a uniform 7×7 grid (Anderson et al., 2018; a big historical boost).
- **Self-critical sequence training (SCST)** — after cross-entropy training, fine-tune with reinforcement learning using CIDEr as the reward. Directly optimizes the metric; standard in strong papers.
- **Fine-tune BLIP/BLIP-2** on your dataset and compare with your from-scratch models — a great "classic vs modern" study.
- **Dense captioning / paragraph captioning** — one caption per region, or multi-sentence descriptions.
- **Multilingual captioning** — e.g., generate Arabic or French captions (interesting tokenization and evaluation challenges).
- **Domain-specific captioning** — medical images (chest X-ray reports) or remote sensing; small specialized datasets, high real-world value.
- **Hallucination study** — measure object hallucination with the CHAIR metric and test mitigation strategies; an active research area directly relevant to modern multimodal LLMs.

---

## Closing Summary

```
See (encoder) → Bridge (attention / cross-attention) → Speak (decoder)
```

Every architecture in this guide is a variation of these three blocks. You will rebuild, in six weeks, the 2014→2021 history of the field — and by doing so, you will understand exactly what is inside today's giant multimodal models. Start small (Flickr8k + CNN+LSTM), make it work end to end, then improve one block at a time, and let the comparison table tell the story.

Good luck — and enjoy the moment your model captions its first photo. It never gets old.


