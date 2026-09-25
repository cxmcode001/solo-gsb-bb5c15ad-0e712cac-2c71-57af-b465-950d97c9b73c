"""已保存方案练习路线规划核心的单元测试。"""

from __future__ import annotations

import random
import time
from itertools import permutations

import pytest

from app.practice_route import (
    MAX_PRACTICE_MEASURES,
    PracticeRouteError,
    plan_practice_route,
)


# ------------------------------------------------------------ 暴力参考实现


def _brute(page_of, measures, start_page, deps):
    """枚举全部排列：最小总翻页距离 + 同成本小节序号序列字典序最小。

    依赖以可比较的小节序号给出；只接受满足全部依赖的排列。
    """
    index_of = {m: i for i, m in enumerate(measures)}
    pairs = [(index_of[a], index_of[b]) for a, b in deps]
    best_cost = None
    best_order = None
    for perm in permutations(range(len(measures))):
        position = [0] * len(measures)
        for pos, idx in enumerate(perm):
            position[idx] = pos
        if any(position[a] >= position[b] for a, b in pairs):
            continue
        order = [measures[idx] for idx in perm]
        cost = abs(start_page - page_of[order[0]])
        for prev, cur in zip(order, order[1:]):
            cost += abs(page_of[prev] - page_of[cur])
        if best_cost is None or cost < best_cost or (
            cost == best_cost and order < best_order
        ):
            best_cost = cost
            best_order = order
    assert best_order is not None, "暴力参考：不存在可行排列"
    return best_order, best_cost


def _make_pages(page_starts, measure_count):
    """由 page_starts 生成每个小节的 1 基页码表。"""
    page_of = [0] * measure_count
    for page, start in enumerate(page_starts, start=1):
        end = page_starts[page] if page < len(page_starts) else measure_count
        for m in range(start, end):
            page_of[m] = page
    return page_of


# ------------------------------------------------------------ 确定性用例


def test_basic_four_pages_start_middle():
    # 8 小节 / 每页 2 小节；目标小节 0,2,4,6 分别在页 1..4，起始页 2
    page_starts = [0, 2, 4, 6]
    route = plan_practice_route(
        page_starts, 8, [0, 2, 4, 6], start_page=2, dependencies=[]
    )
    # 两个方向都可行，从页 2 出发先走页 1（[0,2,4,6] 与 [6,4,2,0] 同为 4），
    # 字典序取小节序号序列最小者
    assert route.order == [0, 2, 4, 6]
    assert route.pages == [1, 2, 3, 4]
    assert route.total_distance == 4
    prev_page = route.start_page
    for seg, measure, page in zip(route.segments, route.order, route.pages):
        assert seg.measure == measure
        assert seg.page == page
        assert seg.from_page == prev_page
        assert seg.distance == abs(page - prev_page)
        prev_page = page
    assert sum(seg.distance for seg in route.segments) == route.total_distance


def test_start_page_to_first_is_counted():
    page_starts = [0, 2, 4, 6]
    route = plan_practice_route(page_starts, 8, [6], start_page=1, dependencies=[])
    assert route.order == [6]
    assert route.pages == [4]
    assert route.total_distance == 3
    assert route.segments[0].from_page == 1
    assert route.segments[0].distance == 3


def test_zero_distance_when_start_on_first_page():
    page_starts = [0]  # 全篇一页
    route = plan_practice_route(page_starts, 3, [0, 1, 2], start_page=1,
                                dependencies=[])
    assert route.order == [0, 1, 2]
    assert route.total_distance == 0
    assert all(seg.distance == 0 for seg in route.segments)


def test_multiple_measures_on_same_page():
    page_starts = [0, 2, 4, 6]
    # 小节 0,1 同页 1；6,7 同页 4
    route = plan_practice_route(
        page_starts, 8, [7, 1, 6, 0], start_page=1, dependencies=[]
    )
    assert route.order == [0, 1, 6, 7]
    assert route.pages == [1, 1, 4, 4]
    assert route.total_distance == 3


def test_lexicographic_tie_break_by_measure_number():
    page_starts = [0, 2, 4, 6]
    # 小节 0 在页 1、小节 4 在页 3，起始页 2：两个顺序总距离均为 3
    route = plan_practice_route(
        page_starts, 8, [4, 0], start_page=2, dependencies=[]
    )
    assert route.order == [0, 4]
    assert route.total_distance == 3


def test_dependency_forces_otherwise_costlier_order():
    page_starts = [0, 2, 4, 6]
    # 无依赖时从页 1 出发 [0,2,4,6] 代价 3；依赖 (6,0) 强制先练 6 后练 0
    route = plan_practice_route(
        page_starts, 8, [0, 2, 4, 6], start_page=1,
        dependencies=[(6, 0)],
    )
    assert route.order == [2, 4, 6, 0]
    assert route.total_distance == 6
    assert route.order.index(6) < route.order.index(0)
    for seg, prev in zip(route.segments, [1] + route.pages[:-1]):
        assert seg.from_page == prev


def test_chain_dependency_single_topological_order():
    page_starts = list(range(15))
    deps = [(i + 1, i) for i in range(14)]  # 14→13→…→0
    route = plan_practice_route(
        page_starts, 15, list(range(15)), start_page=1, dependencies=deps
    )
    assert route.order == list(range(14, -1, -1))


def test_duplicate_dependency_pair_equivalent_to_one():
    page_starts = [0, 2, 4, 6]
    route = plan_practice_route(
        page_starts, 8, [0, 2], start_page=1,
        dependencies=[(0, 2), (0, 2)],
    )
    assert route.order == [0, 2]
    assert route.total_distance == 1


def test_dependency_across_same_page_measures():
    page_starts = [0, 2, 4, 6]
    # 小节 1、0 同在页 1：依赖 (1,0) 只影响顺序、不增加距离
    route = plan_practice_route(
        page_starts, 8, [0, 1], start_page=1, dependencies=[(1, 0)]
    )
    assert route.order == [1, 0]
    assert route.total_distance == 0


# ------------------------------------------------------------ 错误用例


def test_empty_measures_rejected():
    with pytest.raises(PracticeRouteError, match="不能为空"):
        plan_practice_route([0], 1, [], start_page=1, dependencies=[])


def test_too_many_measures_rejected():
    with pytest.raises(PracticeRouteError, match=f"超过上限 {MAX_PRACTICE_MEASURES}"):
        plan_practice_route(
            list(range(16)), 16, list(range(16)), start_page=1, dependencies=[]
        )


def test_boundary_15_accepted():
    route = plan_practice_route(
        list(range(15)), 15, list(range(15)), start_page=1, dependencies=[]
    )
    assert len(route.order) == 15


@pytest.mark.parametrize("measure", [-1, 8, 100])
def test_measure_out_of_range_rejected(measure):
    with pytest.raises(PracticeRouteError, match="越界"):
        plan_practice_route(
            [0, 2, 4, 6], 8, [measure], start_page=1, dependencies=[]
        )


def test_duplicate_measure_rejected():
    with pytest.raises(PracticeRouteError, match="重复"):
        plan_practice_route(
            [0, 2, 4, 6], 8, [0, 2, 0], start_page=1, dependencies=[]
        )


@pytest.mark.parametrize("bad_page", [0, -1, 5])
def test_start_page_out_of_range_rejected(bad_page):
    with pytest.raises(PracticeRouteError, match="起始页码"):
        plan_practice_route(
            [0, 2, 4, 6], 8, [0], start_page=bad_page, dependencies=[]
        )


def test_dependency_pointing_to_non_target_rejected():
    with pytest.raises(PracticeRouteError, match="非目标小节"):
        plan_practice_route(
            [0, 2, 4, 6], 8, [0, 2], start_page=1,
            dependencies=[(0, 5)],
        )
    with pytest.raises(PracticeRouteError, match="非目标小节"):
        plan_practice_route(
            [0, 2, 4, 6], 8, [0, 2], start_page=1,
            dependencies=[(5, 0)],
        )


def test_self_dependency_is_a_cycle():
    with pytest.raises(PracticeRouteError, match="环"):
        plan_practice_route(
            [0, 2, 4, 6], 8, [0], start_page=1, dependencies=[(0, 0)]
        )


def test_dependency_cycle_rejected():
    with pytest.raises(PracticeRouteError, match="环"):
        plan_practice_route(
            [0, 2, 4, 6], 8, [0, 2, 4], start_page=1,
            dependencies=[(0, 2), (2, 4), (4, 0)],
        )


def test_cycle_error_lists_involved_measures():
    # 环仅涉及小节 2、4；环外小节 0、6 不应被列入
    with pytest.raises(PracticeRouteError, match="环") as exc_info:
        plan_practice_route(
            [0, 2, 4, 6], 8, [0, 2, 4, 6], start_page=1,
            dependencies=[(2, 4), (4, 2)],
        )
    message = exc_info.value.args[0]
    assert "2" in message and "4" in message
    # 0、6 两个数字不应出现在环小节列表片段中
    involved = message.split("涉及小节", 1)[1].split("）", 1)[0]
    assert set(involved.replace("、", " ").split()) == {"2", "4"}


@pytest.mark.parametrize("bad_measure", [1.0, 1.5, True, False, "1", None])
def test_non_integer_measure_rejected(bad_measure):
    with pytest.raises(PracticeRouteError, match="整数"):
        plan_practice_route(
            [0, 2, 4, 6], 8, [bad_measure], start_page=1, dependencies=[]
        )


@pytest.mark.parametrize("bad_page", [1.0, 1.5, True, False, "1", None])
def test_non_integer_start_page_rejected(bad_page):
    with pytest.raises(PracticeRouteError, match="整数"):
        plan_practice_route(
            [0, 2, 4, 6], 8, [0], start_page=bad_page, dependencies=[]
        )


@pytest.mark.parametrize(
    "bad_dep",
    [
        (0,),
        (0, 1, 2),
        (0.5, 1),
        (0, True),
        (False, 1),
    ],
)
def test_malformed_dependency_rejected(bad_dep):
    with pytest.raises(PracticeRouteError):
        plan_practice_route(
            [0, 2, 4, 6], 8, [0, 1, 2], start_page=1,
            dependencies=[bad_dep],
        )


# ------------------------------------------------------------ 随机对照


def _random_instance(rng, n, max_pages):
    """生成随机方案分页与目标小节、起始页、依赖 DAG。"""
    # 随机分页：页起始小节为 0..measure_count-1 的递增子序列
    measure_count = n
    breaks = set(rng.sample(range(1, measure_count), min(max_pages - 1,
                                                         measure_count - 1)))
    page_starts = [0] + sorted(breaks)
    measures = list(range(measure_count))
    start_page = rng.randrange(1, len(page_starts) + 1)

    # 随机 DAG：只允许 a < b 的边（按小节序号），保证无环
    edges = []
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < 0.25:
                edges.append((measures[a], measures[b]))
    return page_starts, measure_count, measures, start_page, edges


@pytest.mark.parametrize("seed", range(200))
def test_matches_brute_force(seed):
    rng = random.Random(seed)
    n = rng.randrange(1, 9)
    max_pages = rng.randrange(1, n + 1)
    page_starts, measure_count, measures, start_page, deps = _random_instance(
        rng, n, max_pages
    )
    route = plan_practice_route(
        page_starts, measure_count, measures,
        start_page=start_page, dependencies=deps,
    )
    page_of = _make_pages(page_starts, measure_count)
    exp_order, exp_cost = _brute(page_of, measures, start_page, deps)
    assert route.order == exp_order
    assert route.total_distance == exp_cost
    # 依赖确实满足
    for a, b in deps:
        assert route.order.index(a) < route.order.index(b)
    # 段信息一致
    prev = start_page
    for seg, measure in zip(route.segments, route.order):
        assert seg.measure == measure
        assert seg.page == page_of[measure]
        assert seg.from_page == prev
        assert seg.distance == abs(seg.page - prev)
        prev = seg.page


def test_random_cycles_detected():
    """随机生成含环依赖时，核心逻辑必须报环错误而不是给出错误路线。"""
    rng = random.Random(42)
    cycles_found = 0
    for _ in range(100):
        n = rng.randrange(2, 8)
        measures = list(range(n))
        edges = []
        for a in range(n):
            for b in range(n):
                if a != b and rng.random() < 0.2:
                    edges.append((a, b))
        try:
            plan_practice_route(
                list(range(n)), n, measures, start_page=1, dependencies=edges
            )
            # 无错误时依赖必须无环：Kahn 独立验证
            succ = {m: set() for m in measures}
            for a, b in edges:
                succ[a].add(b)
            seen = set()
            stack = list(succ)
            # 简易 DFS 判环
            color = {m: 0 for m in measures}
            acyclic = True

            def dfs(x):
                nonlocal acyclic
                color[x] = 1
                for y in succ[x]:
                    if color[y] == 1:
                        acyclic = False
                    elif color[y] == 0:
                        dfs(y)
                color[x] = 2

            for m in measures:
                if color[m] == 0:
                    dfs(m)
            assert acyclic
        except PracticeRouteError as exc:
            assert "环" in str(exc)
            cycles_found += 1
    assert cycles_found > 0  # 100 组随机边理应产生若干含环图


def test_performance_15_measures():
    page_starts = list(range(15))
    start = time.monotonic()
    route = plan_practice_route(
        page_starts, 15, list(range(15)), start_page=8, dependencies=[]
    )
    elapsed = time.monotonic() - start
    assert len(route.order) == 15
    assert elapsed < 10.0  # 纯 Python 状态压缩 DP，秒级完成
