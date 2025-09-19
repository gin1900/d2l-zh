"""工具：面向流程型保障效能仿真计算的资源故障概率分析。

该脚本接受一个JSON输入文件，其中描述了仿真的资源及其每个小时的故障率
（例如0.004代表0.4%的故障率），以及多次仿真运行中各资源的使用时长（单位：
小时）。

脚本会对每个资源计算：
* 每次仿真运行期间发生故障的概率；
* 多次仿真综合后至少出现一次故障的概率；
* 期望故障次数；
* 累计使用时长、参与仿真的次数等指标；
并输出为结构化的JSON，方便前端可视化。

用法示例::

    python contrib/failure_rate_simulation.py \
        --input contrib/failure_rate_sample_input.json \
        --output static/failure_rate_sample_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional


def _round_float(value: float, digits: int = 10) -> float:
    """对浮点数进行四舍五入，避免JSON中出现冗长的小数。"""

    return round(float(value), digits)


def _validate_failure_rate(value: float) -> float:
    """Validate that the failure rate is between 0 and 1."""
    if value < 0:
        raise ValueError("资源的故障率必须大于等于0")
    if value >= 1:
        raise ValueError("资源的故障率必须小于1（代表每小时100%故障）")
    return float(value)


@dataclass(frozen=True)
class Resource:
    """Represent a simulated resource with an hourly failure probability."""

    name: str
    failure_rate_per_hour: float

    def __post_init__(self) -> None:
        _validate_failure_rate(self.failure_rate_per_hour)

    @property
    def hazard_rate(self) -> float:
        """Return the continuous-time hazard rate derived from per-hour failure rate.

        我们假设每小时的故障率描述的是离散时间内发生故障的概率，为了支持任意
        时长的使用区间，将其转换为指数分布的瞬时失效率（hazard rate）：

            hazard = -ln(1 - p_hour)
        """

        if self.failure_rate_per_hour == 0:
            return 0.0
        return -math.log1p(-self.failure_rate_per_hour)

    def failure_probability(self, usage_hours: float) -> float:
        """Probability that the resource fails during `usage_hours` of activity."""

        if usage_hours < 0:
            raise ValueError("资源使用时长不能为负数")
        if usage_hours == 0 or self.failure_rate_per_hour == 0:
            return 0.0
        hazard = self.hazard_rate
        return 1.0 - math.exp(-hazard * usage_hours)


@dataclass(frozen=True)
class SimulationRun:
    """Represent a single simulation execution and per-resource usage."""

    run_id: str
    usage_hours: Mapping[str, float]


class ResourceAccumulator:
    """Accumulate run statistics for a resource before producing a report."""

    def __init__(self, resource: Resource) -> None:
        self.resource = resource
        self.total_usage: float = 0.0
        self.exposure_count: int = 0
        self._per_run_entries: List[MutableMapping[str, float]] = []

    def add_run(self, run: SimulationRun) -> None:
        usage_hours = float(run.usage_hours.get(self.resource.name, 0.0))
        if usage_hours < 0:
            raise ValueError(
                f"仿真运行 {run.run_id} 中资源 {self.resource.name} 的使用时长不能为负数"
            )
        if usage_hours == 0:
            return
        failure_probability = self.resource.failure_probability(usage_hours)
        self.total_usage += usage_hours
        self.exposure_count += 1
        self._per_run_entries.append(
            {
                "run_id": run.run_id,
                "usage_hours": usage_hours,
                "failure_probability": failure_probability,
            }
        )

    def build_summary(self) -> Mapping[str, object]:
        if not self._per_run_entries:
            return {
                "name": self.resource.name,
                "failure_rate_per_hour": self.resource.failure_rate_per_hour,
                "total_usage_hours": 0.0,
                "exposure_count": 0,
                "expected_failures": 0.0,
                "overall_failure_probability": 0.0,
                "average_failure_probability": 0.0,
                "per_run": [],
            }

        probabilities = [entry["failure_probability"] for entry in self._per_run_entries]
        expected_failures = float(sum(probabilities))
        survival_product = 1.0
        for probability in probabilities:
            survival_product *= 1.0 - probability
        overall_failure_probability = 1.0 - survival_product
        average_failure_probability = expected_failures / self.exposure_count

        per_run_entries = [
            {
                "run_id": entry["run_id"],
                "usage_hours": _round_float(entry["usage_hours"], 6),
                "failure_probability": _round_float(entry["failure_probability"], 10),
            }
            for entry in self._per_run_entries
        ]

        return {
            "name": self.resource.name,
            "failure_rate_per_hour": self.resource.failure_rate_per_hour,
            "total_usage_hours": _round_float(self.total_usage, 6),
            "exposure_count": self.exposure_count,
            "expected_failures": _round_float(expected_failures, 10),
            "overall_failure_probability": _round_float(
                overall_failure_probability, 10
            ),
            "average_failure_probability": _round_float(
                average_failure_probability, 10
            ),
            "per_run": per_run_entries,
        }


class FailureAnalyzer:
    """Compute failure probabilities across multiple simulation runs."""

    def __init__(self, resources: Iterable[Resource]):
        resource_map = {resource.name: resource for resource in resources}
        if not resource_map:
            raise ValueError("至少需要一个资源来计算故障率")
        self._resource_map = resource_map

    def generate_report(self, runs: Iterable[SimulationRun]) -> Mapping[str, object]:
        accumulators: Dict[str, ResourceAccumulator] = {
            name: ResourceAccumulator(resource)
            for name, resource in self._resource_map.items()
        }
        run_count = 0
        for run in runs:
            run_count += 1
            for accumulator in accumulators.values():
                accumulator.add_run(run)
        if run_count == 0:
            raise ValueError("至少需要一条仿真运行记录")

        resource_summaries = [
            accumulators[name].build_summary() for name in self._resource_map
        ]

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_runs": run_count,
            "resources": resource_summaries,
            "assumptions": {
                "故障率定义": "failure_rate_per_hour 表示每使用一小时发生故障的概率",
                "时间建模": "将离散故障率转换为指数分布失效率，以支持任意时长的使用",
            },
        }


def load_input(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "resources" not in data or "runs" not in data:
        raise ValueError("输入JSON必须包含 resources 和 runs 字段")
    return data


def parse_resources(raw_resources: Iterable[Mapping[str, object]]) -> List[Resource]:
    resources = []
    for item in raw_resources:
        try:
            name = str(item["name"])
            failure_rate = _validate_failure_rate(float(item["failure_rate_per_hour"]))
        except KeyError as exc:
            raise KeyError(f"资源定义缺少字段: {exc}") from exc
        resources.append(Resource(name=name, failure_rate_per_hour=failure_rate))
    return resources


def parse_runs(raw_runs: Iterable[Mapping[str, object]]) -> List[SimulationRun]:
    runs: List[SimulationRun] = []
    for index, item in enumerate(raw_runs, start=1):
        run_id = str(item.get("run_id", f"run_{index:03d}"))
        usage_hours_raw = item.get("usage_hours")
        if not isinstance(usage_hours_raw, Mapping):
            raise TypeError(f"仿真运行 {run_id} 的 usage_hours 必须是对象")
        usage_hours: Dict[str, float] = {}
        for resource_name, value in usage_hours_raw.items():
            usage_hours[str(resource_name)] = float(value)
        runs.append(SimulationRun(run_id=run_id, usage_hours=usage_hours))
    return runs


def build_report(input_path: Path) -> Mapping[str, object]:
    raw_data = load_input(input_path)
    resources = parse_resources(raw_data.get("resources", []))
    runs = parse_runs(raw_data.get("runs", []))
    analyzer = FailureAnalyzer(resources)
    return analyzer.generate_report(runs)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="包含资源定义与仿真记录的JSON文件")
    parser.add_argument("--output", type=Path, help="输出统计结果的JSON路径，缺省时打印到stdout")
    parser.add_argument("--indent", type=int, default=2, help="输出JSON的缩进，默认为2")
    args = parser.parse_args(argv)

    report = build_report(args.input)
    output_text = json.dumps(report, indent=args.indent, ensure_ascii=False)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            f.write(output_text)
            f.write("\n")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
