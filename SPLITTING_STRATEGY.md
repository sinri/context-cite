# Context-Cite 分割策略详解

本文档详细说明 `context-cite` 包中如何将上下文分割成多个源（sources）的策略。

## 概述

`context-cite` 使用 `SimpleContextPartitioner` 类来分割上下文。该分割器支持两种分割模式：
- **句子级别分割** (`source_type="sentence"`): 将上下文按句子分割
- **单词级别分割** (`source_type="word"`): 将上下文按单词分割

## 实现架构

### 类层次结构

```
BaseContextPartitioner (抽象基类)
    └── SimpleContextPartitioner (默认实现)
```

### 核心组件

1. **SimpleContextPartitioner**: 在 `context_partitioner.py` 中实现
2. **split_text()**: 在 `utils.py` 中实现，执行实际的分割逻辑

## 分割策略详解

### 1. 句子级别分割 (`source_type="sentence"`)

#### 实现方式

```python
# 在 utils.py 的 split_text() 函数中
for line in text.splitlines():
    if split_by == "sentence":
        parts.extend(nltk.sent_tokenize(line))
```

#### 分割流程

1. **按行分割**: 首先使用 `text.splitlines()` 将文本按换行符分割成多行
2. **句子分割**: 对每一行使用 NLTK 的 `sent_tokenize()` 函数进行句子分割
3. **合并结果**: 将所有行的句子合并到一个列表中

#### 特点

- **工具**: 使用 NLTK (Natural Language Toolkit) 的 `sent_tokenize` 函数
- **语言**: 默认支持英文句子分割
- **处理换行**: 换行符会被保留在分隔符中，但不会作为句子边界
- **句子边界识别**: NLTK 能够识别常见的句子结束标记（如 `.`, `!`, `?`）

#### 示例

```python
context = "这是第一句话。这是第二句话！\n这是新的一行。"
# 分割结果:
# parts = ["这是第一句话。", "这是第二句话！", "这是新的一行。"]
# separators = ["", " ", "\n"]  # 第一个为空，后续为空格或换行
```

#### 注意事项

- NLTK 需要下载 `punkt_tab` 数据包（代码中已自动下载）
- 对于非英文文本，NLTK 的句子分割可能不够准确
- 跨行的句子不会被合并（每行独立处理）

---

### 2. 单词级别分割 (`source_type="word"`)

#### 实现方式

```python
# 在 utils.py 的 split_text() 函数中
elif split_by == "word":
    tokenizer = English().tokenizer
    parts = [token.text for token in tokenizer(text)]
```

#### 分割流程

1. **创建分词器**: 使用 spaCy 的英文分词器 (`English().tokenizer`)
2. **分词**: 对整个文本进行分词（注意：这里是对整个 `text` 分词，而不是对每行分词）
3. **提取文本**: 提取每个 token 的文本内容

#### 特点

- **工具**: 使用 spaCy 的英文分词器
- **语言**: 默认支持英文单词分割
- **处理方式**: 直接对整个文本进行分词，不考虑换行符
- **分词规则**: spaCy 会处理标点符号、缩写、连字符等复杂情况

#### 示例

```python
context = "The quick brown fox jumps."
# 分割结果:
# parts = ["The", "quick", "brown", "fox", "jumps", "."]
# separators = ["", " ", " ", " ", " ", ""]
```

#### 注意事项

⚠️ **潜在问题**: 当前实现中，`split_text()` 函数在循环 `text.splitlines()` 中，但对 `"word"` 模式使用的是整个 `text` 而不是 `line`。这意味着：
- 代码逻辑上存在不一致（循环了但没用到循环变量）
- 实际效果是对整个文本分词，忽略换行符
- 这可能是一个实现上的小问题，但不影响功能

---

## 分隔符和索引处理

### 分隔符 (Separators)

在分割过程中，系统会记录每个部分之间的分隔符：

1. **第一个部分**: 分隔符为空字符串 `""`
2. **后续部分**: 分隔符为原始文本中该部分前的所有字符（包括空格、换行等）

### 起始索引 (Start Indices)

系统会记录每个部分在原始文本中的起始字符索引，用于后续的字符到 token 的转换。

### 实现逻辑

```python
cur_start = 0
for part in parts:
    cur_end = text.find(part, cur_start)  # 在文本中查找该部分
    separator = text[cur_start:cur_end]   # 提取分隔符
    separators.append(separator)
    start_indices.append(cur_end)         # 记录起始索引
    cur_start = cur_end + len(part)       # 更新当前位置
```

---

## 使用方式

### 默认使用（句子分割）

```python
from context_cite import ContextCiter

citer = ContextCiter.from_pretrained(
    "model-name",
    context="你的上下文文本。包含多个句子。",
    query="你的查询",
    source_type="sentence"  # 默认值
)
```

### 使用单词分割

```python
citer = ContextCiter.from_pretrained(
    "model-name",
    context="你的上下文文本",
    query="你的查询",
    source_type="word"  # 使用单词级别分割
)
```

### 自定义分割器

如果需要更复杂的分割策略，可以实现自定义的 `BaseContextPartitioner`:

```python
from context_cite.context_partitioner import BaseContextPartitioner

class ParagraphPartitioner(BaseContextPartitioner):
    """按段落分割（双换行符）"""
    
    def __init__(self, context: str):
        super().__init__(context)
        self.paragraphs = [p.strip() for p in context.split("\n\n") if p.strip()]
    
    @property
    def num_sources(self) -> int:
        return len(self.paragraphs)
    
    def split_context(self) -> None:
        pass  # 已在 __init__ 中完成
    
    def get_source(self, index: int) -> str:
        return self.paragraphs[index]
    
    def get_context(self, mask=None):
        if mask is None:
            return self.context
        return "\n\n".join([
            self.paragraphs[i] 
            for i in range(len(self.paragraphs)) 
            if mask[i]
        ])

# 使用自定义分割器
citer = ContextCiter.from_pretrained(
    "model-name",
    context="段落1\n\n段落2\n\n段落3",
    query="查询",
    partitioner=ParagraphPartitioner("段落1\n\n段落2\n\n段落3")
)
```

---

## 分割策略的选择

### 何时使用句子分割 (`source_type="sentence"`)

✅ **适合场景**:
- 上下文包含完整的句子
- 需要理解句子级别的语义贡献
- 上下文较长，需要粗粒度的归因分析
- 每个句子是相对独立的信息单元

**优点**:
- 分割粒度适中，不会产生太多源
- 每个源包含完整的语义信息
- 计算效率较高（源数量较少）

**缺点**:
- 无法精确定位到单词级别
- 对于短句子可能信息量不足

### 何时使用单词分割 (`source_type="word"`)

✅ **适合场景**:
- 需要精确到单词级别的归因
- 上下文较短
- 关键词识别很重要
- 需要细粒度的分析

**优点**:
- 可以精确定位到每个单词的贡献
- 适合分析特定关键词的影响

**缺点**:
- 会产生大量源（每个单词一个源）
- 计算成本高（消融实验数量会很大）
- 单词级别的归因可能不够稳定

---

## 性能考虑

### 源数量对性能的影响

- **源数量少** (句子分割): 
  - 消融实验更快
  - 内存占用更少
  - 但归因精度可能较低

- **源数量多** (单词分割):
  - 消融实验更慢
  - 内存占用更大
  - 但归因精度更高

### 建议

- **长文本**: 使用 `source_type="sentence"`
- **短文本**: 可以使用 `source_type="word"`
- **平衡**: 根据具体需求在精度和效率之间权衡

---

## 实现细节和潜在问题

### 1. 句子分割的换行处理

当前实现中，句子分割先按行分割，然后对每行进行句子分割。这意味着：
- 跨行的句子不会被识别为一个完整句子
- 换行符会被保留在分隔符中

### 2. 单词分割的实现不一致

在 `split_text()` 函数中，`"word"` 模式的处理存在逻辑不一致：
- 代码在 `for line in text.splitlines()` 循环中
- 但实际使用的是整个 `text` 而不是 `line`
- 这可能是代码遗留问题，但不影响功能（因为确实是对整个文本分词）

### 3. 非英文文本支持

- **句子分割**: NLTK 的 `sent_tokenize` 主要针对英文优化
- **单词分割**: spaCy 的 `English()` 分词器只支持英文

对于其他语言，建议：
- 实现自定义分割器
- 使用对应语言的 NLP 工具（如中文使用 jieba）

---

## 3. 自定义分隔符分割 (`SeparatorContextPartitioner`)

### 概述

`SeparatorContextPartitioner` 允许用户使用自定义分隔符（字符串或正则表达式）来分割上下文。这是最灵活的分割方式，特别适合有明确结构标记的文本。

### 实现方式

```python
from context_cite.context_partitioner import SeparatorContextPartitioner

# 使用独立行分隔符
partitioner = SeparatorContextPartitioner(
    context="段落1\n====\n段落2\n====\n段落3",
    separator="====",
    match_whole_line=True
)
```

### 分割流程

1. **独立行分隔符模式** (`match_whole_line=True`):
   - 按行分割文本
   - 检查每行是否完全匹配分隔符模式
   - 匹配的行作为分隔符，不包含在源中
   - 非匹配行累积为源内容

2. **正则表达式模式** (`match_whole_line=False`):
   - 使用正则表达式在整个文本中查找分隔符
   - 使用 `re.split()` 分割文本
   - 分隔符被移除，不包含在源中

### 特点

- **灵活性**: 支持任意字符串或正则表达式作为分隔符
- **精确控制**: 可以精确指定分割位置
- **格式保持**: 在重建上下文时保留原始分隔符
- **多语言支持**: 不依赖特定语言的 NLP 工具

### 使用场景

✅ **适合场景**:
- 文本有明确的结构标记（如 `====`、`---` 等）
- 需要按段落、章节等逻辑单元分割
- 文本格式固定，有统一的分隔符
- 需要精确控制分割位置
- 多语言文本（不依赖语言特定的 NLP 工具）

**优点**:
- 完全控制分割逻辑
- 不依赖外部 NLP 库
- 适合结构化文本
- 支持复杂的正则表达式模式

**缺点**:
- 需要文本有明确的分隔符
- 对于无结构标记的文本不适用
- 需要用户了解正则表达式（如果使用复杂模式）

### 示例

#### 示例 1: 独立行分隔符

```python
context = """第一段内容
包含多行文本

====

第二段内容
也是多行

====

第三段内容"""

partitioner = SeparatorContextPartitioner(
    context=context,
    separator="====",
    match_whole_line=True
)

print(partitioner.num_sources)  # 输出: 3
print(partitioner.get_source(0))  # 输出: "第一段内容\n包含多行文本"
```

#### 示例 2: 正则表达式分隔符

```python
context = "源1\n---\n源2\n---\n源3"

partitioner = SeparatorContextPartitioner(
    context=context,
    separator=r"^---+$",  # 匹配独立的横线行（一个或多个横线）
    match_whole_line=True
)
```

#### 示例 3: 简单字符串分隔符

```python
context = "部分1---部分2---部分3"

partitioner = SeparatorContextPartitioner(
    context=context,
    separator="---",
    match_whole_line=False  # 不要求整行匹配
)
```

#### 示例 4: 与 ContextCiter 集成

```python
from context_cite import ContextCiter
from context_cite.context_partitioner import SeparatorContextPartitioner

# 创建自定义分割器
partitioner = SeparatorContextPartitioner(
    context="段落1\n====\n段落2\n====\n段落3",
    separator="====",
    match_whole_line=True
)

# 在 ContextCiter 中使用
citer = ContextCiter.from_pretrained(
    "model-name",
    context="段落1\n====\n段落2\n====\n段落3",
    query="查询",
    partitioner=partitioner  # 使用自定义分割器
)

# 获取归因
attributions = citer.get_attributions(as_dataframe=True, top_k=3)
print(attributions)
```

### 参数说明

- **`separator`**: 
  - 可以是字符串（会被转义为字面量）或已编译的正则表达式
  - 如果 `match_whole_line=True`，字符串会被转换为匹配整行的正则表达式

- **`match_whole_line`**:
  - `True`: 分隔符必须匹配整行（适合独立行标记）
  - `False`: 分隔符可以在行的任何位置（适合行内分隔符）

- **`strip_sources`**:
  - `True`: 自动去除每个源的前后空白字符（推荐）
  - `False`: 保留原始空白字符

### 边界情况处理

- **分隔符在开头**: 第一个源可能为空字符串
- **分隔符在结尾**: 最后一个源可能为空字符串
- **连续分隔符**: 多个连续分隔符之间的空源会被保留
- **无分隔符**: 整个文本作为一个源返回
- **分隔符不匹配**: 返回整个文本作为一个源

### 最佳实践

1. **选择合适的分隔符**:
   - 使用在文本中唯一出现的标记
   - 避免与内容冲突的字符

2. **独立行 vs 行内分隔符**:
   - 如果分隔符是独立的行（如 `====`），使用 `match_whole_line=True`
   - 如果分隔符在行内（如 `---`），使用 `match_whole_line=False`

3. **正则表达式使用**:
   - 简单字符串分隔符会自动转义，无需担心特殊字符
   - 复杂模式可以使用已编译的正则表达式对象

4. **性能考虑**:
   - 正则表达式会被编译并缓存，性能良好
   - 对于长文本，独立行模式可能比正则表达式模式稍快

---

## 总结

当前 `context-cite` 的分割策略：

1. **默认策略**: 句子级别分割 (`source_type="sentence"`)
2. **可选策略**: 单词级别分割 (`source_type="word"`)
3. **自定义分隔符**: 使用 `SeparatorContextPartitioner` 指定任意分隔符
4. **完全自定义**: 实现 `BaseContextPartitioner` 接口

分割策略的选择应该根据：
- 上下文长度
- 需要的归因精度
- 计算资源限制
- 文本结构特征
- 具体应用场景

选择合适的策略可以在归因精度和计算效率之间取得平衡。

### 策略选择指南

| 策略 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| 句子分割 | 长文本，需要句子级别归因 | 粒度适中，效率高 | 无法精确定位单词 |
| 单词分割 | 短文本，需要精确归因 | 精确定位 | 源数量多，计算成本高 |
| 自定义分隔符 | 结构化文本，有明确标记 | 完全控制，灵活 | 需要文本有分隔符 |
| 完全自定义 | 特殊需求，复杂逻辑 | 最大灵活性 | 需要实现接口 |
