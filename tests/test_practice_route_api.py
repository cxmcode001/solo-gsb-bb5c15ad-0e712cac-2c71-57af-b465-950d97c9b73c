"""已保存方案练习路线接口的端到端测试（FastAPI TestClient + 临时 SQLite）。"""

from __future__ import annotations

import importlib
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """以临时 DATA_DIR 重新加载 app.main，得到隔离的服务实例。"""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    sys.modules.pop("app.main", None)
    main = importlib.import_module("app.main")
    with TestClient(main.app) as c:
        yield c


def _create_plan(client, measure_count=8, page_height=2) -> str:
    """创建并保存一个规划：每小节高度 1，容量 page_height，静默充足。"""
    resp = client.post(
        "/api/v1/plans",
        json={
            "measure_heights": [1] * measure_count,
            "silence_seconds": [5] * measure_count,
            "page_height": page_height,
            "min_turn_time": 1,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def test_practice_route_basic(client):
    # 8 小节 / 每页 2 小节 => 4 页；小节 0,2,4,6 在页 1..4，起始页 2
    plan_id = _create_plan(client)
    resp = client.post(
        f"/api/v1/plans/{plan_id}/practice-route",
        json={
            "practice_measures": [0, 2, 4, 6],
            "start_page": 2,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["id"] == plan_id
    assert body["page_count"] == 4
    assert body["start_page"] == 2
    assert body["order"] == [0, 2, 4, 6]
    assert body["pages"] == [1, 2, 3, 4]
    assert body["total_distance"] == 4
    assert body["segments"] == [
        {"measure": 0, "page": 1, "from_page": 2, "distance": 1},
        {"measure": 2, "page": 2, "from_page": 1, "distance": 1},
        {"measure": 4, "page": 3, "from_page": 2, "distance": 1},
        {"measure": 6, "page": 4, "from_page": 3, "distance": 1},
    ]


def test_practice_route_with_dependency(client):
    plan_id = _create_plan(client)
    resp = client.post(
        f"/api/v1/plans/{plan_id}/practice-route",
        json={
            "practice_measures": [0, 2, 4, 6],
            "start_page": 1,
            "dependencies": [[6, 0]],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # 依赖强制小节 6 先于 0
    assert body["order"] == [2, 4, 6, 0]
    assert body["total_distance"] == 6
    assert body["segments"][0]["from_page"] == 1
    assert body["segments"][-1] == {
        "measure": 0, "page": 1, "from_page": 4, "distance": 3
    }


def test_practice_route_dependencies_default_empty(client):
    plan_id = _create_plan(client)
    resp = client.post(
        f"/api/v1/plans/{plan_id}/practice-route",
        json={"practice_measures": [7], "start_page": 4},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["order"] == [7]
    assert body["pages"] == [4]
    assert body["total_distance"] == 0


def test_practice_route_request_echoed(client):
    plan_id = _create_plan(client)
    payload = {
        "practice_measures": [0, 6],
        "start_page": 1,
        "dependencies": [[0, 6]],
    }
    resp = client.post(f"/api/v1/plans/{plan_id}/practice-route", json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()["request"] == payload


def test_practice_route_plan_not_found_returns_404(client):
    resp = client.post(
        "/api/v1/plans/does-not-exist/practice-route",
        json={"practice_measures": [0], "start_page": 1},
    )
    assert resp.status_code == 404
    assert "规划结果不存在" in resp.json()["detail"]


@pytest.mark.parametrize(
    "payload,fragment",
    [
        # 空目标
        ({"practice_measures": [], "start_page": 1}, "不能为空"),
        # 超过 15 个
        (
            {
                "practice_measures": list(range(16)),
                "start_page": 1,
            },
            "超过上限 15",
        ),
        # 小节越界
        ({"practice_measures": [8], "start_page": 1}, "越界"),
        ({"practice_measures": [-1], "start_page": 1}, "越界"),
        # 小节重复
        (
            {"practice_measures": [0, 0], "start_page": 1},
            "重复",
        ),
        # 起始页码越界
        ({"practice_measures": [0], "start_page": 0}, "起始页码"),
        ({"practice_measures": [0], "start_page": 5}, "起始页码"),
        # 依赖指向非目标小节
        (
            {
                "practice_measures": [0, 2],
                "start_page": 1,
                "dependencies": [[0, 5]],
            },
            "非目标小节",
        ),
        # 依赖成环
        (
            {
                "practice_measures": [0, 2],
                "start_page": 1,
                "dependencies": [[0, 2], [2, 0]],
            },
            "环",
        ),
        # 自依赖也是环
        (
            {
                "practice_measures": [0],
                "start_page": 1,
                "dependencies": [[0, 0]],
            },
            "环",
        ),
    ],
)
def test_practice_route_semantic_errors_return_422(client, payload, fragment):
    plan_id = _create_plan(client)
    resp = client.post(f"/api/v1/plans/{plan_id}/practice-route", json=payload)
    assert resp.status_code == 422, resp.text
    assert fragment in resp.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        # 小节序号 / 起始页为浮点或布尔
        {"practice_measures": [1.5], "start_page": 1},
        {"practice_measures": [1.0], "start_page": 1},
        {"practice_measures": [True], "start_page": 1},
        {"practice_measures": [0], "start_page": 1.0},
        {"practice_measures": [0], "start_page": True},
        {"practice_measures": ["0"], "start_page": 1},
        {"practice_measures": [0], "start_page": "1"},
        {"practice_measures": [None], "start_page": 1},
        # 依赖结构或元素类型非法
        {"practice_measures": [0, 1], "start_page": 1, "dependencies": [0]},
        {"practice_measures": [0, 1], "start_page": 1, "dependencies": [[0]]},
        {"practice_measures": [0, 1], "start_page": 1, "dependencies": [[0, 1, 2]]},
        {"practice_measures": [0, 1], "start_page": 1, "dependencies": [[0, 1.0]]},
        {"practice_measures": [0, 1], "start_page": 1, "dependencies": [[True, 1]]},
        {"practice_measures": [0, 1], "start_page": 1, "dependencies": [["0", 1]]},
    ],
)
def test_practice_route_type_errors_return_422(client, payload):
    plan_id = _create_plan(client)
    resp = client.post(f"/api/v1/plans/{plan_id}/practice-route", json=payload)
    assert resp.status_code == 422, resp.text


def test_practice_route_missing_and_extra_fields_rejected(client):
    plan_id = _create_plan(client)
    missing_start = client.post(
        f"/api/v1/plans/{plan_id}/practice-route",
        json={"practice_measures": [0]},
    )
    assert missing_start.status_code == 422
    missing_measures = client.post(
        f"/api/v1/plans/{plan_id}/practice-route", json={"start_page": 1}
    )
    assert missing_measures.status_code == 422
    extra = client.post(
        f"/api/v1/plans/{plan_id}/practice-route",
        json={"practice_measures": [0], "start_page": 1, "note": "x"},
    )
    assert extra.status_code == 422


def test_practice_route_15_measures_upper_bound(client):
    # 15 小节 / 每页 1 小节 => 15 页
    plan_id = _create_plan(client, measure_count=15, page_height=1)
    resp = client.post(
        f"/api/v1/plans/{plan_id}/practice-route",
        json={
            "practice_measures": list(range(15)),
            "start_page": 8,
            "dependencies": [[i, i + 1] for i in range(14)],
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["order"] == list(range(15))
    assert body["pages"] == list(range(1, 16))


def test_practice_route_does_not_modify_saved_plan(client):
    plan_id = _create_plan(client)
    before = client.get(f"/api/v1/plans/{plan_id}").json()
    resp = client.post(
        f"/api/v1/plans/{plan_id}/practice-route",
        json={"practice_measures": [0, 6], "start_page": 2},
    )
    assert resp.status_code == 200
    after = client.get(f"/api/v1/plans/{plan_id}").json()
    assert before == after


def test_practice_route_single_page_plan(client):
    # 3 小节一页装下 => page_count 1，任何路线距离均为 0
    plan_id = _create_plan(client, measure_count=3, page_height=10)
    resp = client.post(
        f"/api/v1/plans/{plan_id}/practice-route",
        json={"practice_measures": [2, 0, 1], "start_page": 1},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["page_count"] == 1
    assert body["order"] == [0, 1, 2]
    assert body["total_distance"] == 0
    assert all(seg["distance"] == 0 for seg in body["segments"])
