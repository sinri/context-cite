# Context-Cite 包 API 文档

本文档详细说明了 `context-cite` 包中所有模块、类和方法的实现细节及参数说明。

## 目录

1. [包概览](#包概览)
2. [context_cite.context_partitioner 模块](#context_citecontext_partitioner-模块)
3. [context_cite.context_citer 模块](#context_citecontext_citer-模块)
4. [context_cite.utils 模块](#context_citeutils-模块)
5. [context_cite.solver 模块](#context_citesolver-模块)

---

## 包概览

`context-cite` 是一个用于将大语言模型（LLM）生成的语句归因（或引用）到上下文信息的工具包。它通过消融实验（ablation）和线性回归模型来量化每个上下文源对模型输出的贡献。

### 主要功能

- 将上下文分割成多个源（sources）
- 通过消融实验评估每个源的重要性
- 使用线性回归模型计算归因分数
- 提供可视化的归因结果

---

## context_cite.context_partitioner 模块

该模块提供了将上下文分割成多个源的抽象接口和实现。

### BaseContextPartitioner（抽象基类）

用于将上下文分割成源的基类。

#### `__init__(self, context: str) -> None`

初始化上下文分割器。

**参数：**
- `context` (str): 待分割的上下文文本

**属性：**
- `context` (str): 存储的上下文文本

#### `num_sources` (property) -> int

**抽象属性**，返回上下文中的源数量。

**返回：**
- `int`: 上下文被分割成的源的数量

#### `split_context(self) -> None`

**抽象方法**，将上下文分割成源。子类必须实现此方法。

#### `get_source(self, index: int) -> str`

**抽象方法**，获取指定索引对应的源。

**参数：**
- `index` (int): 源的索引（从 0 开始）

**返回：**
- `str`: 对应索引的源文本

#### `get_context(self, mask: Optional[NDArray] = None) -> str`

**抽象方法**，根据给定的掩码获取消融后的上下文版本。

**参数：**
- `mask` (Optional[NDArray]): 布尔数组，指定哪些源要保留。如果为 `None`，则返回完整上下文。数组长度应等于源的数量，`True` 表示保留该源，`False` 表示移除该源。

**返回：**
- `str`: 根据掩码过滤后的上下文文本

#### `sources` (property) -> List[str]

返回所有源的列表。

**返回：**
- `List[str]`: 包含所有源文本的列表，按索引顺序排列

---

### SimpleContextPartitioner（简单上下文分割器）

基于分隔符将上下文分割成源的简单实现。

#### 分割策略说明

`SimpleContextPartitioner` 支持两种分割模式：

1. **句子分割** (`source_type="sentence"`, 默认):
   - 使用 NLTK 的 `sent_tokenize()` 函数
   - 首先按换行符分割文本，然后对每一行进行句子分割
   - 每个句子成为一个独立的源
   - 适合长文本和需要句子级别归因的场景

2. **单词分割** (`source_type="word"`):
   - 使用 spaCy 的英文分词器 (`English().tokenizer`)
   - 对整个文本进行分词（不考虑换行符）
   - 每个单词成为一个独立的源
   - 适合需要精确到单词级别的归因分析

**注意**: 两种模式都主要针对英文文本优化。对于其他语言，建议实现自定义分割器。

#### `__init__(self, context: str, source_type: str = "sentence") -> None`

初始化简单上下文分割器。

**参数：**
- `context` (str): 待分割的上下文文本
- `source_type` (str, optional): 源的类型，决定如何分割文本。可选值：
  - `"sentence"`: 按句子分割（默认），使用 NLTK 的句子分词器
  - `"word"`: 按单词分割，使用 spaCy 的英文分词器

**属性：**
- `context` (str): 存储的上下文文本
- `source_type` (str): 源的类型
- `_cache` (dict): 内部缓存，用于存储分割结果（`parts` 和 `separators`）

#### `split_context(self) -> None`

将文本分割成部分并缓存结果。调用 `utils.split_text()` 函数进行实际分割。

**缓存内容：**
- `parts`: 分割后的文本部分列表
- `separators`: 分隔符列表

#### `parts` (property) -> List[str]

获取分割后的文本部分。如果缓存中不存在，会自动调用 `split_context()`。

**返回：**
- `List[str]`: 分割后的文本部分列表

#### `separators` (property) -> List[str]

获取分隔符列表。如果缓存中不存在，会自动调用 `split_context()`。

**返回：**
- `List[str]`: 分隔符列表

#### `num_sources` (property) -> int

返回源的数量，即分割后的部分数量。

**返回：**
- `int`: 源的数量

#### `get_source(self, index: int) -> str`

获取指定索引的源。

**参数：**
- `index` (int): 源的索引

**返回：**
- `str`: 对应索引的源文本

#### `get_context(self, mask: Optional[NDArray] = None) -> str`

根据掩码获取消融后的上下文。

**参数：**
- `mask` (Optional[NDArray]): 布尔数组，指定保留哪些源。如果为 `None`，返回完整上下文。

**返回：**
- `str`: 根据掩码过滤后的上下文，保留的源之间用原始分隔符连接

**实现细节：**
- 使用掩码过滤 `parts` 和 `separators`
- 在相邻部分之间插入相应的分隔符
- 第一个部分前不添加分隔符

---

### SeparatorContextPartitioner（自定义分隔符分割器）

基于自定义分隔符（字符串或正则表达式）将上下文分割成源的实现。支持独立行分隔符（如 `====`）和正则表达式模式。

#### 分割策略说明

`SeparatorContextPartitioner` 支持灵活的分隔符定义：

1. **独立行分隔符** (`match_whole_line=True`):
   - 分隔符必须匹配整行
   - 适合使用独立的行标记（如 `====`、`---`）来分隔段落
   - 分隔符行本身会被移除，不包含在源中

2. **正则表达式分隔符** (`match_whole_line=False`):
   - 使用正则表达式模式匹配分隔符
   - 支持复杂的匹配规则
   - 分隔符本身会被移除

3. **简单字符串分隔符**:
   - 可以指定简单的字符串作为分隔符
   - 自动转义特殊字符（除非使用已编译的正则表达式）

#### `__init__(self, context: str, separator: str | re.Pattern, match_whole_line: bool = False, strip_sources: bool = True) -> None`

初始化自定义分隔符分割器。

**参数：**
- `context` (str): 待分割的上下文文本
- `separator` (str | re.Pattern): 分隔符模式。可以是字符串或已编译的正则表达式模式。如果 `match_whole_line=True`，分隔符会被转换为匹配整行的模式
- `match_whole_line` (bool, optional): 如果为 `True`，分隔符必须匹配整行。这对于独立行分隔符（如 `====`）很有用。默认为 `False`
- `strip_sources` (bool, optional): 如果为 `True`，去除每个源的前后空白字符。默认为 `True`

**属性：**
- `context` (str): 存储的上下文文本
- `separator` (str | re.Pattern): 分隔符模式
- `match_whole_line` (bool): 是否匹配整行
- `strip_sources` (bool): 是否去除源的前后空白
- `_cache` (dict): 内部缓存，用于存储分割结果

**异常：**
- `ValueError`: 如果分隔符模式无效（正则表达式错误）

#### `split_context(self) -> None`

根据分隔符模式分割上下文并缓存结果。

**实现细节：**
- 对于 `match_whole_line=True`：
  - 按行分割文本
  - 检查每行是否匹配分隔符模式
  - 匹配的行作为分隔符，不包含在源中
  - 非匹配行累积为源内容
- 对于 `match_whole_line=False`：
  - 使用 `re.split()` 分割文本
  - 分隔符被移除，不包含在源中
  - 处理文本开头/结尾的分隔符情况

**缓存内容：**
- `parts`: 分割后的文本部分列表（不包含分隔符）
- `separators`: 分隔符列表（用于重建上下文）

#### `parts` (property) -> List[str]

获取分割后的文本部分。如果缓存中不存在，会自动调用 `split_context()`。

**返回：**
- `List[str]`: 分割后的文本部分列表

#### `separators` (property) -> List[str]

获取分隔符列表。如果缓存中不存在，会自动调用 `split_context()`。

**返回：**
- `List[str]`: 分隔符列表，每个元素对应一个源之前的分隔符（第一个源的分隔符通常为空字符串）

#### `num_sources` (property) -> int

返回源的数量，即分割后的部分数量。

**返回：**
- `int`: 源的数量

#### `get_source(self, index: int) -> str`

获取指定索引的源。

**参数：**
- `index` (int): 源的索引（从 0 开始）

**返回：**
- `str`: 对应索引的源文本（不包含分隔符）

#### `get_context(self, mask: Optional[NDArray] = None) -> str`

根据掩码获取消融后的上下文。

**参数：**
- `mask` (Optional[NDArray]): 布尔数组，指定保留哪些源。如果为 `None`，返回完整上下文。数组长度应等于源的数量，`True` 表示保留该源，`False` 表示移除该源。

**返回：**
- `str`: 根据掩码过滤后的上下文，保留的源之间用原始分隔符连接

**异常：**
- `ValueError`: 如果掩码长度与源数量不匹配

**实现细节：**
- 使用掩码过滤 `parts` 和 `separators`
- 在相邻部分之间插入相应的分隔符
- 第一个部分前不添加分隔符（除非第一个源被移除）

#### 使用示例

```python
from context_cite.context_partitioner import SeparatorContextPartitioner

# 使用独立行分隔符
partitioner = SeparatorContextPartitioner(
    context="段落1\n====\n段落2\n====\n段落3",
    separator="====",
    match_whole_line=True
)
print(partitioner.num_sources)  # 输出: 3
print(partitioner.get_source(0))  # 输出: "段落1"

# 使用正则表达式分隔符
partitioner = SeparatorContextPartitioner(
    context="源1\n---\n源2\n---\n源3",
    separator=r"^---+$",  # 匹配独立的横线行
    match_whole_line=True
)

# 使用简单字符串分隔符（非整行）
partitioner = SeparatorContextPartitioner(
    context="部分1---部分2---部分3",
    separator="---",
    match_whole_line=False
)

# 使用已编译的正则表达式
import re
pattern = re.compile(r"SEPARATOR", re.IGNORECASE)
partitioner = SeparatorContextPartitioner(
    context="文本1SEPARATOR文本2",
    separator=pattern,
    match_whole_line=False
)
```

---

## context_cite.context_citer 模块

该模块提供了 `ContextCiter` 类，这是包的核心类，用于执行上下文归因分析。

### ContextCiter 类

用于将模型生成的响应归因到上下文中的各个源。

#### `__init__(self, model: Any, tokenizer: Any, context: str, query: str, source_type: str = "sentence", generate_kwargs: Optional[Dict[str, Any]] = None, num_ablations: int = 64, ablation_keep_prob: float = 0.5, batch_size: int = 1, solver: Optional[BaseSolver] = None, prompt_template: str = DEFAULT_PROMPT_TEMPLATE, partitioner: Optional[BaseContextPartitioner] = None) -> None`

初始化 ContextCiter 实例。

**参数：**
- `model` (Any): 要应用 ContextCite 的模型（HuggingFace ModelForCausalLM 实例）
- `tokenizer` (Any): 与模型关联的分词器
- `context` (str): 提供给模型的上下文
- `query` (str): 向模型提出的查询
- `source_type` (str, optional): 将上下文分割成源的类型。默认为 `"sentence"`，也可以是 `"word"`
- `generate_kwargs` (Optional[Dict[str, Any]], optional): 传递给模型 `generate()` 方法的额外关键字参数。默认为 `{"max_new_tokens": 512, "do_sample": False}`
- `num_ablations` (int, optional): 用于训练代理模型的消融实验数量。默认为 `64`
- `ablation_keep_prob` (float, optional): 消融上下文时保留源的概率。默认为 `0.5`
- `batch_size` (int, optional): 使用消融上下文进行推理时的批次大小。默认为 `1`
- `solver` (Optional[BaseSolver], optional): 用于计算线性代理模型的求解器。默认使用 Lasso 回归
- `prompt_template` (str, optional): 用于从上下文和查询创建提示的模板字符串。默认为 `"Context: {context}\n\nQuery: {query}"`
- `partitioner` (Optional[BaseContextPartitioner], optional): 用于将上下文分割成源的自定义分割器。如果指定，将覆盖 `source_type` 参数

**属性：**
- `model`: 存储的模型实例
- `tokenizer`: 存储的分词器实例
- `partitioner`: 上下文分割器实例
- `query`: 存储的查询文本
- `generate_kwargs`: 生成参数
- `num_ablations`: 消融实验数量
- `ablation_keep_prob`: 保留概率
- `batch_size`: 批次大小
- `solver`: 求解器实例
- `prompt_template`: 提示模板
- `_cache`: 内部缓存字典
- `logger`: 日志记录器

**注意事项：**
- 如果分词器没有 `pad_token`，会自动设置为 `eos_token`
- 如果提供了自定义 `partitioner`，会验证其上下文是否与提供的 `context` 匹配

---

#### `from_pretrained(cls, pretrained_model_name_or_path, context: str, query: str, device: str = "cuda", model_kwargs: Dict[str, Any] = {}, tokenizer_kwargs: Dict[str, Any] = {}, **kwargs: Dict[str, Any]) -> "ContextCiter"`

**类方法**，从预训练模型加载 ContextCiter 实例。

**参数：**
- `pretrained_model_name_or_path` (str): 预训练模型的名称或路径。可以是本地路径或 HuggingFace 模型中心的模型名称
- `context` (str): 提供给模型的上下文
- `query` (str): 提供给模型的查询
- `device` (str, optional): 使用的设备。默认为 `"cuda"`
- `model_kwargs` (Dict[str, Any], optional): 传递给模型构造函数的额外关键字参数。默认为空字典
- `tokenizer_kwargs` (Dict[str, Any], optional): 传递给分词器构造函数的额外关键字参数。默认为空字典
- `**kwargs` (Dict[str, Any], optional): 传递给 ContextCiter 构造函数的额外关键字参数

**返回：**
- `ContextCiter`: 使用提供的模型、分词器、上下文、查询和其他关键字参数初始化的 ContextCiter 实例

**实现细节：**
- 使用 `AutoModelForCausalLM.from_pretrained()` 加载模型
- 使用 `AutoTokenizer.from_pretrained()` 加载分词器
- 将模型移动到指定设备
- 设置分词器的 `padding_side` 为 `"left"`

---

#### `_get_prompt_ids(self, mask: Optional[NDArray] = None, return_prompt: bool = False)`

**私有方法**，获取提示的 token ID。

**参数：**
- `mask` (Optional[NDArray]): 用于过滤上下文的掩码。如果为 `None`，使用完整上下文
- `return_prompt` (bool, optional): 是否同时返回文本提示。默认为 `False`

**返回：**
- 如果 `return_prompt=False`: 返回 `List[int]`，提示的 token ID 列表
- 如果 `return_prompt=True`: 返回 `Tuple[List[int], str]`，包含 token ID 列表和文本提示

**实现细节：**
- 使用 `partitioner.get_context(mask)` 获取（可能消融后的）上下文
- 使用 `prompt_template.format()` 格式化提示
- 应用聊天模板（如果分词器支持）
- 编码为 token ID

---

#### `_response_start` (property) -> int

**私有属性**，返回响应开始的 token 索引。

**返回：**
- `int`: 响应在完整输出中开始的 token 索引位置

---

#### `_output` (property) -> str

**私有属性**，返回模型的完整输出（包括提示和响应）。结果会被缓存。

**返回：**
- `str`: 模型的完整输出文本

**实现细节：**
- 如果缓存中不存在，会生成新的输出
- 使用 `_get_prompt_ids(return_prompt=True)` 获取提示
- 调用 `model.generate()` 生成响应
- 解码并组合提示和响应

---

#### `_output_tokens` (property) -> dict

**私有属性**，返回完整输出的 tokenization 结果。

**返回：**
- `dict`: 包含 `input_ids` 等键的字典，由分词器返回

---

#### `_response_ids` (property) -> List[int]

**私有属性**，返回响应的 token ID 列表（不包括提示部分）。

**返回：**
- `List[int]`: 响应的 token ID 列表

---

#### `response` (property) -> str

返回模型生成的响应（不包括提示部分）。此属性会被缓存。

**返回：**
- `str`: 模型生成的响应文本

**实现细节：**
- 使用 `_output_tokens.token_to_chars()` 找到响应开始的字符位置
- 从完整输出中提取响应部分

---

#### `response_with_indices(self, split_by="word", color=True) -> [str, pd.DataFrame]`

返回模型生成的响应，并标注每个部分的起始索引。

**参数：**
- `split_by` (str, optional): 分割响应的方法。可以是 `"word"` 或 `"sentence"`。默认为 `"word"`
- `color` (bool, optional): 是否对每个部分的起始索引进行颜色高亮。默认为 `True`

**返回：**
- `str`: 带有起始索引标注的响应文本（使用 ANSI 颜色代码高亮）

**实现细节：**
- 使用 `utils.split_text()` 分割响应
- 使用 `utils.highlight_word_indices()` 添加索引标注和颜色

---

#### `num_sources` (property) -> int

返回上下文中的源数量。

**返回：**
- `int`: 上下文被分割成的源的数量

---

#### `sources` (property) -> List[str]`

返回上下文中的所有源。

**返回：**
- `List[str]`: 上下文中的源列表，每个元素是一个源

---

#### `_char_range_to_token_range(self, start_index, end_index) -> Tuple[int, int]`

**私有方法**，将字符范围转换为 token 范围。

**参数：**
- `start_index` (int): 起始字符索引
- `end_index` (int): 结束字符索引（不包含）

**返回：**
- `Tuple[int, int]`: `(token_start, token_end)`，相对于响应开始的 token 索引范围

**实现细节：**
- 考虑响应在完整输出中的偏移量
- 使用 `utils.char_to_token()` 进行转换

---

#### `_indices_to_token_indices(self, start_index=None, end_index=None) -> Tuple[int, int]`

**私有方法**，将字符索引转换为 token 索引，并进行验证。

**参数：**
- `start_index` (Optional[int]): 起始字符索引。如果为 `None`，默认为 0
- `end_index` (Optional[int]): 结束字符索引。如果为 `None`，默认为响应长度

**返回：**
- `Tuple[int, int]`: `(token_start, token_end)`，相对于响应开始的 token 索引范围

**异常：**
- `ValueError`: 如果索引范围无效（不在 0 到响应长度之间）

---

#### `_compute_masks_and_logit_probs(self) -> None`

**私有方法**，计算消融掩码和对数概率。结果会被缓存。

**实现细节：**
- 调用 `utils.get_masks_and_logit_probs()` 生成掩码和计算对数概率
- 将结果存储在 `_cache` 中，键为 `"reg_masks"` 和 `"reg_logit_probs"`

---

#### `_masks` (property) -> NDArray`

**私有属性**，返回消融掩码数组。如果缓存中不存在，会自动计算。

**返回：**
- `NDArray`: 形状为 `(num_ablations, num_sources)` 的布尔数组，每行表示一次消融实验保留哪些源

---

#### `_logit_probs` (property) -> NDArray`

**私有属性**，返回对数概率数组。如果缓存中不存在，会自动计算。

**返回：**
- `NDArray`: 形状为 `(num_ablations, response_length)` 的数组，包含每次消融实验对每个响应 token 的对数概率

---

#### `_get_attributions_for_ids_range(self, ids_start_idx, ids_end_idx) -> Tuple[NDArray, float]`

**私有方法**，计算指定 token 范围（相对于响应开始）的归因分数。

**参数：**
- `ids_start_idx` (int): 起始 token 索引（相对于响应开始）
- `ids_end_idx` (int): 结束 token 索引（相对于响应开始，不包含）

**返回：**
- `Tuple[NDArray, float]`: `(weight, bias)`，其中：
  - `weight`: 形状为 `(num_sources,)` 的数组，每个元素是对应源的归因分数
  - `bias`: 偏置项（标量）

**实现细节：**
- 使用 `utils.aggregate_logit_probs()` 聚合指定范围的 token 对数概率
- 使用 `solver.fit()` 拟合线性模型，得到每个源的权重

---

#### `get_attributions(self, start_idx: Optional[int] = None, end_idx: Optional[int] = None, as_dataframe: bool = False, top_k: Optional[int] = None, verbose: bool = True) -> NDArray | Any`

获取（部分）响应的归因分数。

**参数：**
- `start_idx` (Optional[int]): 要归因的响应部分的起始字符索引。如果为 `None`，默认为响应的开始
- `end_idx` (Optional[int]): 要归因的响应部分的结束字符索引。如果为 `None`，默认为响应的结束
- `as_dataframe` (bool, optional): 如果为 `True`，返回格式化的 DataFrame（按分数排序）。否则，返回 numpy 数组，其中第 i 个元素对应第 i 个源的分数。默认为 `False`
- `top_k` (Optional[int], optional): 仅在 `as_dataframe=True` 时使用。返回前 k 个归因结果。如果为 `None`，返回所有归因结果。默认为 `None`
- `verbose` (bool, optional): 如果为 `True`，打印被归因的响应部分。默认为 `True`

**返回：**
- 如果 `as_dataframe=False`: 返回 `NDArray`，形状为 `(num_sources,)`，第 i 个元素对应第 i 个源的分数
- 如果 `as_dataframe=True`: 返回格式化的 pandas DataFrame，包含 `Score` 和 `Source` 列，按分数降序排列，带有颜色高亮

**实现细节：**
- 将字符索引转换为 token 索引
- 验证选择的文本与解码的 token 是否匹配（如果不匹配会发出警告）
- 调用 `_get_attributions_for_ids_range()` 计算归因
- 如果 `as_dataframe=True`，使用 `utils.get_attributions_df()` 格式化结果

**注意事项：**
- 如果 `num_sources == 0`，返回空数组并发出警告
- 如果 `top_k` 不为 `None` 但 `as_dataframe=False`，会发出警告并忽略 `top_k`

---

## context_cite.utils 模块

该模块提供了各种工具函数，用于文本处理、消融实验和归因计算。

### `split_text(text: str, split_by: str) -> Tuple[List[str], List[str], List[str]]`

将文本分割成部分，并返回部分、起始索引和分隔符。

**参数：**
- `text` (str): 要分割的文本
- `split_by` (str): 分割方式，可以是：
  - `"sentence"`: 按句子分割（使用 NLTK 的 `sent_tokenize`）
  - `"word"`: 按单词分割（使用 spaCy 的英文分词器）

**返回：**
- `Tuple[List[str], List[str], List[str]]`: `(parts, separators, start_indices)`，其中：
  - `parts`: 分割后的文本部分列表
  - `separators`: 每个部分前的分隔符列表（第一个部分的分隔符通常为空字符串）
  - `start_indices`: 每个部分在原文中的起始字符索引列表

**异常：**
- `ValueError`: 如果 `split_by` 不是 `"sentence"` 或 `"word"`

**实现细节：**

**句子分割模式** (`split_by="sentence"`):
1. 使用 `text.splitlines()` 按换行符分割文本
2. 对每一行使用 NLTK 的 `sent_tokenize()` 进行句子分割
3. 将所有行的句子合并到一个列表中
4. 换行符会被保留在分隔符中

**单词分割模式** (`split_by="word"`):
1. 创建 spaCy 的英文分词器 (`English().tokenizer`)
2. 对整个文本（而不是逐行）进行分词
3. 提取每个 token 的文本内容
4. 注意：虽然代码在循环中，但实际是对整个文本分词

**分隔符和索引计算**:
- 通过 `text.find(part, cur_start)` 在原文中查找每个部分的位置
- 分隔符是每个部分前的所有字符（包括空格、换行等）
- 起始索引记录每个部分在原文中的字符位置

---

### `highlight_word_indices(words, indices, separators, color: bool) -> str`

为单词添加起始索引标注，并可选择性地添加颜色高亮。

**参数：**
- `words` (List[str]): 单词列表
- `indices` (List[int]): 每个单词的起始索引列表
- `separators` (List[str]): 每个单词前的分隔符列表
- `color` (bool): 是否使用 ANSI 颜色代码高亮索引

**返回：**
- `str`: 格式化的字符串，每个单词前带有其起始索引（用方括号括起来）

**实现细节：**
- 如果 `color=True`，使用青色（`\033[36m`）高亮索引
- 格式为：`[索引]单词`
- 在单词之间插入相应的分隔符

---

### `_create_mask(num_sources, alpha, seed) -> NDArray`

**私有函数**，创建一个随机掩码。

**参数：**
- `num_sources` (int): 源的数量
- `alpha` (float): 保留源的概率（0 到 1 之间）
- `seed` (int): 随机种子

**返回：**
- `NDArray`: 形状为 `(num_sources,)` 的布尔数组，`True` 表示保留该源，`False` 表示移除

**实现细节：**
- 使用 `numpy.random.RandomState` 确保可重复性
- 每个源以概率 `alpha` 被保留

---

### `_create_regression_dataset(num_masks, num_sources, get_prompt_ids, response_ids, alpha, base_seed=0) -> Tuple[NDArray, Dataset]`

**私有函数**，创建用于回归的数据集。

**参数：**
- `num_masks` (int): 要创建的掩码数量
- `num_sources` (int): 源的数量
- `get_prompt_ids` (Callable): 接受掩码并返回提示 token ID 的函数
- `response_ids` (List[int]): 响应的 token ID 列表
- `alpha` (float): 保留源的概率
- `base_seed` (int, optional): 基础随机种子。默认为 0

**返回：**
- `Tuple[NDArray, Dataset]`: `(masks, dataset)`，其中：
  - `masks`: 形状为 `(num_masks, num_sources)` 的布尔数组
  - `dataset`: HuggingFace Dataset，包含 `input_ids`、`attention_mask` 和 `labels`

**实现细节：**
- 为每个掩码创建输入序列（提示 + 响应）
- `labels` 中，提示部分的 token 标记为 `-100`（在损失计算中被忽略），响应部分的 token 使用实际 ID

---

### `_compute_logit_probs(logits, labels) -> NDArray`

**私有函数**，计算每个 token 的对数概率。

**参数：**
- `logits` (torch.Tensor): 形状为 `(batch_size, seq_length, vocab_size)` 的 logits 张量
- `labels` (torch.Tensor): 形状为 `(batch_size, seq_length)` 的标签张量

**返回：**
- `NDArray`: 形状为 `(batch_size, seq_length)` 的数组，包含每个位置的对数概率

**实现细节：**
- 对每个位置，计算正确 token 的 logit 与其他所有 token 的 log-sum-exp 的差值
- 这给出了正确 token 相对于其他 token 的对数概率

---

### `_make_loader(dataset, tokenizer, batch_size) -> DataLoader`

**私有函数**，创建数据加载器。

**参数：**
- `dataset` (Dataset): HuggingFace Dataset
- `tokenizer` (Any): 分词器
- `batch_size` (int): 批次大小

**返回：**
- `DataLoader`: PyTorch DataLoader

**实现细节：**
- 使用 `DataCollatorForSeq2Seq` 进行填充和整理
- 填充到批次中最长序列的长度

---

### `_get_response_logit_probs(dataset, model, tokenizer, response_length, batch_size) -> NDArray`

**私有函数**，计算数据集中所有样本的响应对数概率。

**参数：**
- `dataset` (Dataset): HuggingFace Dataset
- `model` (Any): 模型实例
- `tokenizer` (Any): 分词器
- `response_length` (int): 响应的长度（token 数）
- `batch_size` (int): 批次大小

**返回：**
- `NDArray`: 形状为 `(len(dataset), response_length)` 的数组，包含每个样本每个响应 token 的对数概率

**实现细节：**
- 使用数据加载器批量处理
- 使用 `torch.no_grad()` 和 `torch.cuda.amp.autocast()` 进行推理优化
- 只计算响应部分的 logits（不包括提示部分）
- 使用 `_compute_logit_probs()` 计算对数概率

**注意事项：**
- 如果 `batch_size > 1`，要求分词器使用左填充（`padding_side == "left"`）

---

### `get_masks_and_logit_probs(model, tokenizer, num_masks, num_sources, get_prompt_ids, response_ids, ablation_keep_prob, batch_size, base_seed=0) -> Tuple[NDArray, NDArray]`

生成消融掩码并计算对应的对数概率。

**参数：**
- `model` (Any): 模型实例
- `tokenizer` (Any): 分词器
- `num_masks` (int): 要生成的掩码数量
- `num_sources` (int): 源的数量
- `get_prompt_ids` (Callable): 接受掩码并返回提示 token ID 的函数
- `response_ids` (List[int]): 响应的 token ID 列表
- `ablation_keep_prob` (float): 保留源的概率
- `batch_size` (int): 批次大小
- `base_seed` (int, optional): 基础随机种子。默认为 0

**返回：**
- `Tuple[NDArray, NDArray]`: `(masks, logit_probs)`，其中：
  - `masks`: 形状为 `(num_masks, num_sources)` 的布尔数组
  - `logit_probs`: 形状为 `(num_masks, response_length)` 的数组，数据类型为 `float32`

**实现细节：**
- 调用 `_create_regression_dataset()` 创建数据集和掩码
- 调用 `_get_response_logit_probs()` 计算对数概率

---

### `aggregate_logit_probs(logit_probs, output_type="logit_prob") -> NDArray`

从 token 级别的对数概率计算序列级别的输出。

**参数：**
- `logit_probs` (NDArray): 形状为 `(num_samples, num_tokens)` 的对数概率数组
- `output_type` (str, optional): 聚合类型。可选值：
  - `"logit_prob"`: 对数概率（默认）
  - `"log_prob"`: 对数概率（另一种形式）
  - `"total_token_logit_prob"`: 平均 token 对数概率

**返回：**
- `NDArray`: 形状为 `(num_samples,)` 的数组，包含每个样本的聚合值

**实现细节：**
- `"log_prob"`: 对每个 token 应用 log-sigmoid，然后求和
- `"logit_prob"`: 在 log-prob 的基础上转换为 logit-prob 形式
- `"total_token_logit_prob"`: 对所有 token 的对数概率求平均

**异常：**
- `ValueError`: 如果 `output_type` 不是支持的值

---

### `_color_scale(val, max_val) -> str`

**私有函数**，根据值的大小生成颜色样式字符串。

**参数：**
- `val` (float): 当前值
- `max_val` (float): 最大值

**返回：**
- `str`: CSS 样式字符串，格式为 `"background-color: rgb(r, g, b)"`

**实现细节：**
- 从白色 `(255, 255, 255)` 到绿色 `(80, 180, 80)` 的线性插值
- 值越大，颜色越绿

---

### `_apply_color_scale(df) -> Styler`

**私有函数**，对 DataFrame 的 `Score` 列应用颜色缩放。

**参数：**
- `df` (pd.DataFrame): 包含 `Score` 列的 DataFrame

**返回：**
- `pd.Styler`: 应用了颜色样式的 DataFrame Styler

**实现细节：**
- 最大值的上限为 `max(df["Score"].max(), np.log(10))`
- 分数 `np.log(10)` 表示消融该源会导致对数概率下降 `np.log(10)`，大约对应概率下降 10 倍

---

### `get_attributions_df(attributions: NDArray[Any], context_partitioner, top_k: Optional[int] = None) -> Any`

将归因分数转换为格式化的 DataFrame。

**参数：**
- `attributions` (NDArray): 形状为 `(num_sources,)` 的归因分数数组
- `context_partitioner` (BaseContextPartitioner): 上下文分割器实例
- `top_k` (Optional[int], optional): 返回前 k 个归因结果。如果为 `None`，返回所有结果。默认为 `None`

**返回：**
- `pd.Styler`: 格式化的 DataFrame Styler，包含 `Score` 和 `Source` 列，按分数降序排列，带有颜色高亮和 3 位小数精度

**实现细节：**
- 按分数降序排序
- 如果指定了 `top_k`，只保留前 k 个
- 使用 `context_partitioner.get_source()` 获取每个源的文本
- 应用颜色缩放和格式化

---

### `char_to_token(output_tokens, char_index) -> int`

将字符索引转换为 token 索引。这是为了解决某些模型（如 Llama 3）的 `char_to_token` 方法的 bug。

**参数：**
- `output_tokens` (Any): 分词器的输出（包含 `token_to_chars` 方法）
- `char_index` (int): 字符索引

**返回：**
- `int`: 对应的 token 索引

**实现细节：**
- 遍历所有 token，找到第一个起始字符位置大于 `char_index` 的 token
- 返回该 token 的前一个索引

---

## context_cite.solver 模块

该模块提供了用于拟合线性代理模型的求解器。

### BaseSolver（抽象基类）

求解器的基类。

#### `fit(self, masks: NDArray, outputs: NDArray, num_output_tokens: int) -> Tuple[NDArray, NDArray]`

**抽象方法**，拟合求解器到给定数据。

**参数：**
- `masks` (NDArray): 形状为 `(num_samples, num_sources)` 的布尔数组，表示每次消融实验保留哪些源
- `outputs` (NDArray): 形状为 `(num_samples,)` 的数组，表示每次消融实验的聚合输出
- `num_output_tokens` (int): 输出 token 的数量

**返回：**
- `Tuple[NDArray, NDArray]`: `(weight, bias)`，其中：
  - `weight`: 形状为 `(num_sources,)` 的数组，每个元素是对应源的权重
  - `bias`: 偏置项（标量）

---

### LassoRegression（LASSO 回归求解器）

使用 scikit-learn 库的 LASSO 回归求解器。

#### `__init__(self, lasso_alpha: float = 0.01) -> None`

初始化 LASSO 回归求解器。

**参数：**
- `lasso_alpha` (float, optional): LASSO 回归的 alpha 参数（正则化强度）。默认为 `0.01`

**属性：**
- `lasso_alpha` (float): LASSO 正则化参数

---

#### `fit(self, masks: NDArray, outputs: NDArray, num_output_tokens: int) -> Tuple[NDArray, NDArray]`

拟合 LASSO 回归模型。

**参数：**
- `masks` (NDArray): 形状为 `(num_samples, num_sources)` 的布尔数组
- `outputs` (NDArray): 形状为 `(num_samples,)` 的数组
- `num_output_tokens` (int): 输出 token 的数量

**返回：**
- `Tuple[NDArray, NDArray]`: `(weight, bias)`，其中权重和偏置都已乘以 `num_output_tokens` 以恢复到原始尺度

**实现细节：**
- 将掩码转换为 `float32` 类型
- 将输出除以 `num_output_tokens` 进行归一化
- 使用 `StandardScaler` 标准化输入特征
- 使用 `Lasso` 回归拟合模型（`fit_intercept=True`）
- 将权重和偏置重新缩放到原始尺度（乘以 `num_output_tokens`）
- 调整偏置项以考虑标准化的影响

**数学公式：**
- 标准化后的模型：`Y_norm = (X - mean) / scale @ coef + intercept`
- 恢复后的模型：`Y = X @ (coef / scale * num_output_tokens) + (intercept - mean/scale @ coef) * num_output_tokens`

---

## 使用示例

### 基本用法

```python
from context_cite import ContextCiter

# 从预训练模型创建实例
citer = ContextCiter.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    context="这是上下文文本。它包含多个句子。",
    query="这是什么？",
    device="cuda"
)

# 获取响应
response = citer.response
print(response)

# 获取归因分数（numpy 数组）
attributions = citer.get_attributions()
print(attributions)

# 获取归因分数（DataFrame，前 5 个）
attributions_df = citer.get_attributions(as_dataframe=True, top_k=5)
print(attributions_df)
```

### 自定义参数

```python
citer = ContextCiter.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    context="长上下文文本...",
    query="查询",
    num_ablations=128,  # 增加消融实验数量
    ablation_keep_prob=0.6,  # 调整保留概率
    batch_size=4,  # 增加批次大小
    source_type="word",  # 按单词分割
    generate_kwargs={"max_new_tokens": 256, "temperature": 0.7}
)
```

### 自定义分割器

```python
from context_cite.context_partitioner import BaseContextPartitioner

class CustomPartitioner(BaseContextPartitioner):
    def __init__(self, context: str):
        super().__init__(context)
        self.sources_list = context.split("\n\n")  # 按双换行分割
    
    @property
    def num_sources(self) -> int:
        return len(self.sources_list)
    
    def split_context(self) -> None:
        pass  # 已在 __init__ 中完成
    
    def get_source(self, index: int) -> str:
        return self.sources_list[index]
    
    def get_context(self, mask=None):
        if mask is None:
            return self.context
        return "\n\n".join([self.sources_list[i] for i in range(len(self.sources_list)) if mask[i]])

citer = ContextCiter.from_pretrained(
    "model-name",
    context="上下文",
    query="查询",
    partitioner=CustomPartitioner("上下文")
)
```

---

## 注意事项

1. **性能考虑**：
   - `num_ablations` 越大，计算时间越长，但结果可能更准确
   - `batch_size` 越大，内存占用越大，但可能加快计算速度
   - 首次调用 `get_attributions()` 时会进行所有消融实验，可能需要较长时间

2. **内存使用**：
   - 消融实验的结果会被缓存，可能占用较多内存
   - 对于很长的上下文或响应，可能需要调整批次大小

3. **分词器要求**：
   - 如果使用 `batch_size > 1`，分词器必须使用左填充（`padding_side="left"`）
   - ContextCiter 会自动设置 `pad_token` 如果不存在

4. **模型要求**：
   - 模型必须是 HuggingFace 的 `ModelForCausalLM` 类型
   - 模型必须支持 `generate()` 方法

5. **归因分数解释**：
   - 分数越高，表示该源对生成该响应部分的贡献越大
   - 分数为 `np.log(10)` 表示移除该源会导致概率下降约 10 倍

---

## 版本信息

- 包版本：0.0.4（根据 pyproject.toml）
- Python 要求：>= 3.8
- 主要依赖：numpy, torch, transformers, datasets, pandas, nltk, spacy, scikit-learn, tqdm
