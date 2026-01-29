# todo: 写一个class，提供将给定的 report.json 转换为csv文件的能力
import json
import csv
import os
import re




class ReportToCsv:
    def __init__(self, report_json_path, csv_output_path):
        self.report_json_path = report_json_path
        self.csv_output_path = csv_output_path

    def convert(self):
        with open(self.report_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        with open(self.csv_output_path, "w", encoding="gbk", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "group_index", "ref_rank", "source_index", "score", "source_head", "formatted", "normalized", "contexted"])
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                for q, groups in entry.items():
                    question = (q or "").strip()
                    if not isinstance(groups, list):
                        continue
                    for gi, refs in enumerate(groups):
                        if not isinstance(refs, list):
                            continue
                        for ri, ref in enumerate(refs):
                            if not isinstance(ref, dict):
                                continue
                            source_head = ref.get("source_head")
                            formatted, normalized, contexted = self.__source_head_to_flags(source_head)
                            writer.writerow([
                                question,
                                gi,
                                ri,
                                ref.get("index"),
                                ref.get("score"),
                                source_head,
                                formatted,
                                normalized,
                                contexted,
                            ])

    def __source_head_to_flags(self,source_head):
        """
        从 source_head（如 "Reference [3]"）解析数字 n，按 exp_case_builder 的位定义返回
        (formatted, normalized, contexted)，值为 1 或 0。无法解析时返回 ("", "", "")。
        """
        if source_head is None:
            return "", "", ""
        m = re.search(r"\[(\d+)\]", str(source_head).strip())
        if not m:
            return "", "", ""
        n = int(m.group(1))
        formatted = 1 if (n & 0b001) else 0
        normalized = 1 if (n & 0b010) else 0
        contexted = 1 if (n & 0b100) else 0
        return formatted, normalized, contexted

class AnalyzeReport:
    """
    分析report.csv文件
    """
    def __init__(self, csv_input_path):
        self.csv_input_path = csv_input_path
    
    def _iter_rows(self):
        with open(self.csv_input_path, "r", encoding="gbk", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row
    
    def list_best(self):
        """
        找出 ref_rank 为 0 的行，按照 source_head 的值归口统计行数，按行数倒序排序后返回 [(source_head, count), ...]
        """
        counts = {}
        for row in self._iter_rows():
            try:
                ref_rank = int(str(row.get("ref_rank", "")).strip())
            except ValueError:
                continue
            if ref_rank != 0:
                continue
            source_head = str(row.get("source_head", "")).strip()
            if source_head not in counts:
                counts[source_head] = 0
            counts[source_head] += 1
        return sorted(counts.items(), key=lambda x: (-x[1], x[0]))

    def sum_scores(self):
        """
        按照 source_head 的值归口,计算相关行的 score 总和，返回 [(source_head, total_score), ...]
        """
        totals = {}
        for row in self._iter_rows():
            source_head = str(row.get("source_head", "")).strip()
            try:
                score = float(str(row.get("score", "")).strip())
            except ValueError:
                continue
            if source_head not in totals:
                totals[source_head] = 0.0
            totals[source_head] += score
        return sorted(totals.items(), key=lambda x: (-x[1], x[0]))

    def feature_analyze(self):
        """
        仅取 ref_rank==0 的行，统计每个特征为 1 与为 0 的条数，计算并返回各特征为 1 与为 0 的比例 (proportion_1, proportion_0)。
        返回 {"formatted": (p1, p0), "normalized": (p1, p0), "contexted": (p1, p0)}，二者之和为 1；无数据时返回空 dict 或 total 为 0 时该特征为 (0.0, 0.0)。
        """
        feature_names = ["formatted", "normalized", "contexted"]
        counts = {name: [0, 0] for name in feature_names}  # [count_0, count_1]
        for row in self._iter_rows():
            try:
                ref_rank = int(str(row.get("ref_rank", "")).strip())
            except ValueError:
                continue
            if ref_rank != 0:
                continue
            flags = []
            for name in feature_names:
                raw = str(row.get(name, "")).strip()
                if raw == "1":
                    flags.append(1)
                elif raw == "0":
                    flags.append(0)
                else:
                    flags = None
                    break
            if flags is None:
                continue
            for name, v in zip(feature_names, flags):
                counts[name][v] += 1
        result = {}
        for name in feature_names:
            c0, c1 = counts[name]
            total = c0 + c1
            if total == 0:
                result[name] = (0.0, 0.0)
            else:
                result[name] = (c1 / total, c0 / total)
        return result

if __name__ == "__main__":
    case_code='b'

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    case_dir = os.path.join(base_dir, "data", "internal", "t", case_code)
    report_json_path = os.path.join(case_dir,"report.json")
    csv_output_path = os.path.join(case_dir,"report.csv")
    converter = ReportToCsv(report_json_path, csv_output_path)
    converter.convert()
    analyzer = AnalyzeReport(csv_output_path)
    top_refs = analyzer.list_best()
    total_scores = analyzer.sum_scores()
    print("# Top refs by source_head (ref_rank==0):")
    for k, v in top_refs:
        print(k, v)
    print("----")
    print("# Total scores by source_head:")
    for k, v in total_scores:
        print(k, v)
    print("----")
    feature_result = analyzer.feature_analyze()
    print("# Feature proportions (ref_rank==0, proportion_1, proportion_0):")
    for name, (p1, p0) in feature_result.items():
        print(name,":", p1,"against", p0)
