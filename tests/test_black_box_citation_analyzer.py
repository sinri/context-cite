"""
单元测试：BlackBoxCitationAnalyzer 四参数必填，校验 influence 相对大小。
"""
import os

from context_cite.black_box.black_box_citation_analyzer import (
    BlackBoxCitationAnalyzer,
    BlackBoxCitationResult,
)

# 与 cuda_analyzer_test 一致，可通过环境变量覆盖
_model_source = os.environ.get(
    "CONTEXT_CITE_MODEL_SOURCE",
    "E:\\sinri\\DeepSeek-R1-Distill-Qwen-1.5B",
)


def test_black_box_result_api():
    """BlackBoxCitationResult 必填字段与 getter。"""
    result = BlackBoxCitationResult(
        influence_on_output_1=1.0,
        influence_on_output_2=2.0,
    )
    assert result.get_influence_on_output_1() == 1.0
    assert result.get_influence_on_output_2() == 2.0


def test_black_box_analyzer_constructor():
    """BlackBoxCitationAnalyzer 构造函数与 analyze 四参数必填。"""
    analyzer = BlackBoxCitationAnalyzer(model_source=_model_source)
    # 仅校验存在且可调用，不加载模型
    assert hasattr(analyzer, "analyze")
    import inspect

    sig = inspect.signature(analyzer.analyze)
    params = list(sig.parameters.keys())
    assert params == ["self", "query", "target_text", "output_1", "output_2"]


def test_analyze_signature_and_result():
    """四参数必填，返回 BlackBoxCitationResult，influence_on_output_2 >= influence_on_output_1。"""
    query = "What is the capital of France?"
    target_text = "Paris is the capital and largest city of France."
    output_1 = "I'm not sure."  # 无 target_text 时的典型短回答
    output_2 = "Paris is the capital of France."  # 有 target_text 时的回答

    analyzer = BlackBoxCitationAnalyzer(model_source=_model_source)
    result = analyzer.analyze(
        query=query,
        target_text=target_text,
        output_1=output_1,
        output_2=output_2,
    )

    assert isinstance(result, BlackBoxCitationResult)
    influence_1 = result.get_influence_on_output_1()
    influence_2 = result.get_influence_on_output_2()
    assert isinstance(influence_1, float)
    assert isinstance(influence_2, float)
    # 有 target_text 时生成的 output_2 应对 target_text 的归因更高
    assert influence_2 >= influence_1, (
        f"Expected influence_on_output_2 ({influence_2}) >= influence_on_output_1 ({influence_1})"
    )


if __name__ == "__main__":
    test_black_box_result_api()
    test_black_box_analyzer_constructor()
    print("API tests passed.")
    if os.path.exists(_model_source):
        test_analyze_signature_and_result()
        print("test_analyze_signature_and_result passed.")
    else:
        print(
            f"Skip test_analyze_signature_and_result (model not found at {_model_source})."
        )
