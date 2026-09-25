"""已保存方案的练习路线规划核心。

背景
====
一份已保存的翻页方案把乐谱分成若干页（页码 1..page_count，每页对应一个
连续小节区间）。练习者希望针对 1～15 个**互异的常练小节**安排一次练习的
先后路线：从给定的**起始页码**出发，依次练到每个常练小节；另可提交若干
"先练小节 → 后练小节"的**先后依赖**（二元组）。各常练小节按方案映射到
其所在页，每段路程的代价为**页码距离** |a − b|（起始页到首项也计一段）。

目标是在满足全部依赖的前提下，求一条把每个常练小节恰好练一次的路线，使
各段页码距离之和最小；总距离并列时取**小节序号序列字典序最小**者。

依赖可能指向非目标小节或彼此成环：指向非目标小节直接判非法；环（含
``(m, m)`` 自依赖）使任何拓扑顺序都不存在，同样判非法。

算法
====
``n ≤ 15``，采用状态压缩动态规划。常练小节按序号升序编号为 0..n-1，
依赖预处理为每个小节的"先修小节掩码" ``pred[i]``；先以 Kahn 拓扑剔除
判定依赖是否成环。

后缀代价 DP：``g[mask][i]`` 表示已练小节集合为 ``mask``、当前位于小节
``i``（``i ∈ mask``）时，练完剩余小节所需的最小新增页码距离：

    g[full][*] = 0，
    g[mask][i] = min |page(i) − page(j)| + g[mask∪{j}][j]，
                  j：j∉mask 且 pred[j] ⊆ mask。

首项单独从起始页计入：总距离为
``min |start_page − page(j)| + g[{j}][j]``（要求 ``pred[j] = ∅``）。
随后自空集合起贪心重构：每一步在仍能保持全局最优距离的候选项中取
**小节序号最小**者，即得总距离最小且小节序号序列字典序最小的练习顺序。

复杂度 O(n²·2ⁿ)，n ≤ 15 时约数百万次整数运算。本模块仅依赖标准库，
便于脱离 Web 层直接单测。
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

#: 一次提交的常练小节数上限（状态压缩 DP 的掩码宽度）
MAX_PRACTICE_MEASURES = 15

#: 不可达状态的距离上界
INF = 10**18


class PracticeRouteError(ValueError):
    """练习路线输入不合法：小节/页码越界、重复小节、依赖指向非目标或成环。"""


@dataclass(frozen=True)
class RouteSegment:
    """练习路线中的一段：从某页出发到达某个练习小节。"""

    measure: int  # 本段终点常练小节序号（原谱 0 基）
    page: int  # 该小节所在页（1 基）
    from_page: int  # 本段起点页（首项为起始页，其余为上一项所在页）
    distance: int  # 本段翻页距离（页码差的绝对值，非负）


@dataclass(frozen=True)
class PracticeRoute:
    """练习路线规划结果。"""

    order: List[int]  # 常练小节的练习顺序（原谱 0 基）
    pages: List[int]  # 与 order 一一对应的所在页码（1 基）
    start_page: int  # 起始页码（1 基）
    segments: List[RouteSegment]  # 与 order 一一对应的各段翻页信息
    total_distance: int  # 各段页码距离之和（全局最小）


def _is_strict_int(value: object) -> bool:
    """严格整数：布尔不算（与 Web 层 pydantic 校验口径一致）。"""
    return isinstance(value, int) and not isinstance(value, bool)


def _validate(
    page_starts: Sequence[int],
    measure_count: int,
    practice_measures: Sequence[int],
    start_page: int,
    dependencies: Sequence[Sequence[int]],
) -> Tuple[List[int], List[int], List[int]]:
    """校验输入并返回排序后的常练小节、各小节所在页与先修掩码。

    :raises PracticeRouteError: 任一约束不满足
    """
    n_pages = len(page_starts)

    if not _is_strict_int(start_page):
        raise PracticeRouteError(f"起始页码必须为整数（收到 {start_page!r}）")
    if start_page < 1 or start_page > n_pages:
        raise PracticeRouteError(
            f"起始页码 {start_page} 越界：合法范围为 1..{n_pages}"
        )

    if not isinstance(practice_measures, (list, tuple)):
        raise PracticeRouteError(
            f"practice_measures 必须是数组（收到 {type(practice_measures).__name__}）"
        )
    if not practice_measures:
        raise PracticeRouteError(
            "practice_measures 不能为空：至少需要一个常练小节"
            f"（至多 {MAX_PRACTICE_MEASURES} 个）"
        )
    if len(practice_measures) > MAX_PRACTICE_MEASURES:
        raise PracticeRouteError(
            f"常练小节数（{len(practice_measures)}）超过上限 "
            f"{MAX_PRACTICE_MEASURES}，至多提交 {MAX_PRACTICE_MEASURES} 个互异小节"
        )

    seen: set[int] = set()
    for idx, measure in enumerate(practice_measures):
        if not _is_strict_int(measure):
            raise PracticeRouteError(
                f"第 {idx + 1} 个常练小节序号必须为整数（收到 {measure!r}）"
            )
        if measure < 0 or measure >= measure_count:
            raise PracticeRouteError(
                f"第 {idx + 1} 个常练小节序号 {measure} 越界："
                f"合法范围为 0..{measure_count - 1}"
            )
        if measure in seen:
            raise PracticeRouteError(
                f"常练小节序号 {measure} 重复，常练小节必须互异"
            )
        seen.add(measure)

    measures = sorted(seen)
    index_of = {m: i for i, m in enumerate(measures)}
    # 小节序号 -> 页码：page_starts 升序，bisect_left(page_starts, m+1) 恰为
    # 起始小节 ≤ m 的页数，也就是该小节所在页的 1 基页码。
    pages = [bisect_left(page_starts, m + 1) for m in measures]

    if not isinstance(dependencies, (list, tuple)):
        raise PracticeRouteError(
            f"dependencies 必须是数组（收到 {type(dependencies).__name__}）"
        )

    n = len(measures)
    pred = [0] * n
    succ: List[List[int]] = [[] for _ in range(n)]
    dedup_pairs: set[Tuple[int, int]] = set()
    for idx, dep in enumerate(dependencies):
        if not isinstance(dep, (list, tuple)) or len(dep) != 2:
            raise PracticeRouteError(
                f"第 {idx + 1} 个依赖必须是 [先练小节, 后练小节] 二元组"
                f"（收到 {dep!r}）"
            )
        before, after = dep[0], dep[1]
        if not _is_strict_int(before) or not _is_strict_int(after):
            raise PracticeRouteError(
                f"第 {idx + 1} 个依赖的小节序号必须为整数"
                f"（收到 {dep!r}）"
            )
        if before not in index_of or after not in index_of:
            raise PracticeRouteError(
                f"第 {idx + 1} 个依赖（{before}, {after}）指向非目标小节："
                "依赖两端必须是本次提交的常练小节"
            )
        pair = (before, after)
        if pair in dedup_pairs:
            continue  # 重复依赖等价于一条，不报错
        dedup_pairs.add(pair)
        i, j = index_of[before], index_of[after]
        pred[j] |= 1 << i
        succ[i].append(j)

    # Kahn 拓扑剔除：最终未被剔除的小节全部在某个环上（含自依赖）。
    indegree = [mask.bit_count() for mask in pred]
    ready = 0
    for i, deg in enumerate(indegree):
        if deg == 0:
            ready |= 1 << i
    visited = 0
    while ready:
        bit = ready & -ready
        ready ^= bit
        visited |= bit
        i = bit.bit_length() - 1
        for j in succ[i]:
            indegree[j] -= 1
            if indegree[j] == 0:
                ready |= 1 << j
    if visited != (1 << n) - 1:
        cycle_nodes = [measures[i] for i in range(n) if not (visited >> i) & 1]
        raise PracticeRouteError(
            "依赖关系中存在环（涉及小节 "
            + "、".join(str(m) for m in cycle_nodes)
            + "），无法排出满足全部先后依赖的练习顺序"
        )

    return measures, pages, pred


def plan_practice_route(
    page_starts: Sequence[int],
    measure_count: int,
    practice_measures: Sequence[int],
    start_page: int,
    dependencies: Sequence[Tuple[int, int]],
) -> PracticeRoute:
    """为已保存翻页方案规划一条常练小节练习路线。

    :param page_starts: 每页起始小节（0 基）的升序列表，长度即规划页数，
        页码为其下标 +1
    :param measure_count: 规划小节总数（小节序号范围 0..measure_count-1）
    :param practice_measures: 1～15 个互异的常练小节序号（0 基）
    :param start_page: 起始页码（1 基，合法范围 1..页数）
    :param dependencies: 若干 ``(先练小节, 后练小节)`` 先后依赖，两端均须
        为本次提交的常练小节；重复依赖等价于一条
    :return: 字典序最小的最优练习顺序、各小节所在页、各段翻页距离与总距离
    :raises PracticeRouteError: 小节/页码越界、小节重复、依赖指向非目标
        小节或依赖成环
    """
    measures, pages, pred = _validate(
        page_starts, measure_count, practice_measures, start_page, dependencies
    )
    n = len(measures)
    full = (1 << n) - 1

    # g[(mask, i)]（扁平下标 mask*n+i）：已练集合为 mask、当前在 i 时，
    # 练完剩余小节的最小新增页码距离；不可达为 INF。
    g = [INF] * ((1 << n) * n)
    full_base = full * n
    for i in range(n):
        g[full_base + i] = 0

    # 小节间页码距离矩阵；offset[j] 把 (mask ∪ {j}, j) 的下标折算为
    # mask*n + offset[j]（掩码位互不相交，加法即按位或）。
    dist = [[abs(pages[i] - pages[j]) for j in range(n)] for i in range(n)]
    offset = [(1 << j) * n + j for j in range(n)]

    # 超集值先就绪：按 mask 数值递减即按超集先于子集计算。
    for mask in range(full - 1, 0, -1):
        # 先修已全部练完的未练小节
        candidates = 0
        todo = full ^ mask
        while todo:
            bit = todo & -todo
            j = bit.bit_length() - 1
            if pred[j] & ~mask == 0:
                candidates |= bit
            todo ^= bit

        base = mask * n
        visited_bits = mask
        while visited_bits:
            bit_i = visited_bits & -visited_bits
            i = bit_i.bit_length() - 1
            di = dist[i]
            best = INF
            cand = candidates
            while cand:
                bit_j = cand & -cand
                j = bit_j.bit_length() - 1
                value = di[j] + g[base + offset[j]]
                if value < best:
                    best = value
                cand ^= bit_j
            g[base + i] = best
            visited_bits ^= bit_i

    # 首项：无先修的小节才能作为第一项，距离从起始页起算。
    total_distance: Optional[int] = None
    first = -1
    for j in range(n):
        if pred[j] != 0:
            continue
        value = abs(start_page - pages[j]) + g[offset[j]]
        if total_distance is None or value < total_distance:
            total_distance = value
            first = j
    assert total_distance is not None, "内部错误：练习路线重构失败"

    # 贪心重构：每一步在保持全局最优距离的候选中取小节序号最小者
    # （小节按序号升序编号，故即编号最小），得到字典序最小序列。
    order_measures: List[int] = []
    order_pages: List[int] = []
    segments: List[RouteSegment] = []

    distance = abs(start_page - pages[first])
    segments.append(
        RouteSegment(
            measure=measures[first],
            page=pages[first],
            from_page=start_page,
            distance=distance,
        )
    )
    order_measures.append(measures[first])
    order_pages.append(pages[first])
    mask = 1 << first
    i = first

    while mask != full:
        target = g[mask * n + i]
        chosen = -1
        base = mask * n
        for j in range(n):
            bit_j = 1 << j
            if mask & bit_j or pred[j] & ~mask:
                continue  # 已练或尚有先修未练
            value = dist[i][j] + g[base + offset[j]]
            if value == target:
                chosen = j
                break
        assert chosen >= 0, "内部错误：练习路线重构失败"
        distance = dist[i][chosen]
        segments.append(
            RouteSegment(
                measure=measures[chosen],
                page=pages[chosen],
                from_page=pages[i],
                distance=distance,
            )
        )
        order_measures.append(measures[chosen])
        order_pages.append(pages[chosen])
        mask |= 1 << chosen
        i = chosen

    # 防御：重算总距离必须与 DP 最优值一致，且每小节各出现一次。
    recomputed = sum(seg.distance for seg in segments)
    assert recomputed == total_distance and sorted(order_measures) == measures

    return PracticeRoute(
        order=order_measures,
        pages=order_pages,
        start_page=start_page,
        segments=segments,
        total_distance=total_distance,
    )
