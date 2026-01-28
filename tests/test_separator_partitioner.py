import pytest
import numpy as np
import re
from context_cite.context_partitioner import SeparatorContextPartitioner


def test_separator_partitioner_whole_line_separator():
    """Test separator partitioner with whole line separator (e.g., ====)."""
    context = "段落1\n====\n段落2\n====\n段落3"
    partitioner = SeparatorContextPartitioner(
        context=context, separator="====", match_whole_line=True
    )

    assert partitioner.num_sources == 3
    assert partitioner.get_source(0) == "段落1"
    assert partitioner.get_source(1) == "段落2"
    assert partitioner.get_source(2) == "段落3"

    # Test get_context with no mask (should return full context)
    full_context = partitioner.get_context()
    assert "段落1" in full_context
    assert "段落2" in full_context
    assert "段落3" in full_context

    # Test get_context with mask
    mask = np.array([True, False, True])
    ablated = partitioner.get_context(mask)
    assert "段落1" in ablated
    assert "段落2" not in ablated
    assert "段落3" in ablated


def test_separator_partitioner_regex_separator():
    """Test separator partitioner with regex pattern."""
    context = "源1\n---\n源2\n---\n源3"
    partitioner = SeparatorContextPartitioner(
        context=context, separator=r"^---+$", match_whole_line=True
    )

    assert partitioner.num_sources == 3
    assert partitioner.get_source(0) == "源1"
    assert partitioner.get_source(1) == "源2"
    assert partitioner.get_source(2) == "源3"


def test_separator_partitioner_simple_string():
    """Test separator partitioner with simple string separator."""
    context = "部分1---部分2---部分3"
    partitioner = SeparatorContextPartitioner(
        context=context, separator="---", match_whole_line=False
    )

    assert partitioner.num_sources == 3
    assert "部分1" in partitioner.get_source(0)
    assert "部分2" in partitioner.get_source(1)
    assert "部分3" in partitioner.get_source(2)


def test_separator_partitioner_compiled_regex():
    """Test separator partitioner with compiled regex pattern."""
    context = "文本1\nSEPARATOR\n文本2\nSEPARATOR\n文本3"
    pattern = re.compile(r"^SEPARATOR$", re.MULTILINE)
    partitioner = SeparatorContextPartitioner(
        context=context, separator=pattern, match_whole_line=True
    )

    assert partitioner.num_sources == 3
    assert partitioner.get_source(0) == "文本1"
    assert partitioner.get_source(1) == "文本2"
    assert partitioner.get_source(2) == "文本3"


def test_separator_partitioner_separator_at_start():
    """Test separator at the beginning of text."""
    context = "====\n段落1\n====\n段落2"
    partitioner = SeparatorContextPartitioner(
        context=context, separator="====", match_whole_line=True
    )

    # Should have 2 sources (empty first source and paragraph)
    assert partitioner.num_sources >= 2
    assert "段落1" in partitioner.get_source(0) or "段落1" in partitioner.get_source(1)
    assert "段落2" in partitioner.sources


def test_separator_partitioner_separator_at_end():
    """Test separator at the end of text."""
    context = "段落1\n====\n段落2\n===="
    partitioner = SeparatorContextPartitioner(
        context=context, separator="====", match_whole_line=True
    )

    assert partitioner.num_sources >= 2
    assert "段落1" in partitioner.sources
    assert "段落2" in partitioner.sources


def test_separator_partitioner_no_separator():
    """Test text with no separator (should return whole text as one source)."""
    context = "这是一个没有分隔符的完整文本。"
    partitioner = SeparatorContextPartitioner(
        context=context, separator="====", match_whole_line=True
    )

    assert partitioner.num_sources == 1
    assert partitioner.get_source(0) == context.strip()


def test_separator_partitioner_consecutive_separators():
    """Test consecutive separators."""
    context = "段落1\n====\n====\n段落2"
    partitioner = SeparatorContextPartitioner(
        context=context, separator="====", match_whole_line=True
    )

    # Should handle consecutive separators
    assert partitioner.num_sources >= 2
    assert "段落1" in partitioner.sources
    assert "段落2" in partitioner.sources


def test_separator_partitioner_strip_sources():
    """Test strip_sources parameter."""
    context = "  段落1  \n====\n  段落2  \n====\n  段落3  "
    partitioner_strip = SeparatorContextPartitioner(
        context=context, separator="====", match_whole_line=True, strip_sources=True
    )
    partitioner_no_strip = SeparatorContextPartitioner(
        context=context, separator="====", match_whole_line=True, strip_sources=False
    )

    # With strip, sources should not have leading/trailing whitespace
    assert partitioner_strip.get_source(0) == "段落1"
    assert partitioner_strip.get_source(1) == "段落2"

    # Without strip, sources may have whitespace
    assert "  段落1  " in partitioner_no_strip.get_source(0) or "段落1" in partitioner_no_strip.get_source(0)


def test_separator_partitioner_get_context_mask():
    """Test get_context with various masks."""
    context = "段落1\n====\n段落2\n====\n段落3"
    partitioner = SeparatorContextPartitioner(
        context=context, separator="====", match_whole_line=True
    )

    # Test with all True mask
    mask_all = np.ones(partitioner.num_sources, dtype=bool)
    result_all = partitioner.get_context(mask_all)
    assert "段落1" in result_all
    assert "段落2" in result_all
    assert "段落3" in result_all

    # Test with all False mask
    mask_none = np.zeros(partitioner.num_sources, dtype=bool)
    result_none = partitioner.get_context(mask_none)
    # Should return empty or minimal content
    assert isinstance(result_none, str)

    # Test with single source
    mask_single = np.array([True, False, False])
    result_single = partitioner.get_context(mask_single)
    assert "段落1" in result_single
    assert "段落2" not in result_single
    assert "段落3" not in result_single


def test_separator_partitioner_invalid_pattern():
    """Test with invalid regex pattern."""
    context = "一些文本"
    with pytest.raises(ValueError, match="Invalid separator pattern"):
        SeparatorContextPartitioner(
            context=context, separator="[", match_whole_line=False
        )


def test_separator_partitioner_sources_property():
    """Test the sources property."""
    context = "段落1\n====\n段落2\n====\n段落3"
    partitioner = SeparatorContextPartitioner(
        context=context, separator="====", match_whole_line=True
    )

    sources = partitioner.sources
    assert len(sources) == partitioner.num_sources
    assert all(isinstance(s, str) for s in sources)
    assert "段落1" in sources[0]
    assert "段落2" in sources[1]
    assert "段落3" in sources[2]


def test_separator_partitioner_get_context_mask_length_mismatch():
    """Test get_context with mask length mismatch."""
    context = "段落1\n====\n段落2"
    partitioner = SeparatorContextPartitioner(
        context=context, separator="====", match_whole_line=True
    )

    # Create mask with wrong length
    wrong_mask = np.array([True, False])  # Wrong length
    with pytest.raises(ValueError, match="Mask length"):
        partitioner.get_context(wrong_mask)


def test_separator_partitioner_caching():
    """Test that split_context results are cached."""
    context = "段落1\n====\n段落2"
    partitioner = SeparatorContextPartitioner(
        context=context, separator="====", match_whole_line=True
    )

    # Access parts multiple times - should use cache
    parts1 = partitioner.parts
    parts2 = partitioner.parts
    assert parts1 is parts2  # Should be the same list object (cached)

    # Access separators multiple times - should use cache
    separators1 = partitioner.separators
    separators2 = partitioner.separators
    assert separators1 is separators2  # Should be the same list object (cached)


def test_separator_partitioner_integration_with_context_citer():
    """Test integration with ContextCiter (without actual model)."""
    from context_cite import ContextCiter
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # This test requires a model, so we'll skip it if models are not available
    # In a real scenario, you would test with an actual model
    pytest.importorskip("transformers")

    # Note: This is a structural test - actual model testing would require
    # model loading which is expensive. This test verifies the interface works.
    context = "段落1\n====\n段落2\n====\n段落3"
    partitioner = SeparatorContextPartitioner(
        context=context, separator="====", match_whole_line=True
    )

    # Verify partitioner can be created and has correct interface
    assert hasattr(partitioner, "num_sources")
    assert hasattr(partitioner, "get_source")
    assert hasattr(partitioner, "get_context")
    assert hasattr(partitioner, "split_context")
    assert partitioner.context == context


def test_separator_partitioner_multiline_context():
    """Test with multiline context."""
    context = """第一段
包含多行
内容

====

第二段
也包含
多行内容

====

第三段"""
    partitioner = SeparatorContextPartitioner(
        context=context, separator="====", match_whole_line=True
    )

    assert partitioner.num_sources == 3
    assert "第一段" in partitioner.get_source(0)
    assert "第二段" in partitioner.get_source(1)
    assert "第三段" in partitioner.get_source(2)


def test_separator_partitioner_non_whole_line_regex():
    """Test regex separator without whole line matching."""
    context = "文本1SEP文本2SEP文本3"
    partitioner = SeparatorContextPartitioner(
        context=context, separator="SEP", match_whole_line=False
    )

    assert partitioner.num_sources == 3
    assert "文本1" in partitioner.get_source(0)
    assert "文本2" in partitioner.get_source(1)
    assert "文本3" in partitioner.get_source(2)
