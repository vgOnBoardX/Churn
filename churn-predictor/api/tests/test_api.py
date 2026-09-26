"""API test suite — health, predict, batch, customers, and 422 validation."""
from __future__ import annotations

import io
import pytest


# ── Health ────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_health_returns_ok(client):
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


@pytest.mark.anyio
async def test_root_redirect(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert "service" in r.json()


# ── Predict — valid input ─────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_predict_valid_input(client, valid_customer):
    r = await client.post("/predict", json=valid_customer)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_tier"] in ("low", "medium", "high")
    assert isinstance(body["top_factors"], list)
    assert len(body["top_factors"]) <= 3


@pytest.mark.anyio
async def test_predict_includes_shap_factors(client, valid_customer):
    r = await client.post("/predict", json=valid_customer)
    assert r.status_code == 200
    factors = r.json()["top_factors"]
    for f in factors:
        assert "feature" in f
        assert "shap_impact" in f
        assert f["direction"] in ("increases_churn", "decreases_churn")


# ── Predict — 422 validation ──────────────────────────────────────────────────

@pytest.mark.anyio
async def test_predict_missing_required_field_returns_422(client):
    """Omit tenure_months — must get 422."""
    r = await client.post("/predict", json={
        "monthly_charges": 65.0,
        "total_charges": 1560.0,
        "contract_type": "month-to-month",
        "payment_method": "electronic_check",
        "support_tickets_30d": 1,
    })
    assert r.status_code == 422
    detail = r.json()["detail"]
    fields = [err["loc"] for err in detail]
    assert any("tenure_months" in loc for loc in fields)


@pytest.mark.anyio
async def test_predict_negative_monthly_charges_returns_422(client, valid_customer):
    bad = {**valid_customer, "monthly_charges": -10.0}
    r = await client.post("/predict", json=bad)
    assert r.status_code == 422


@pytest.mark.anyio
async def test_predict_invalid_contract_type_returns_422(client, valid_customer):
    bad = {**valid_customer, "contract_type": "weekly"}
    r = await client.post("/predict", json=bad)
    assert r.status_code == 422


@pytest.mark.anyio
async def test_predict_string_tenure_returns_422(client, valid_customer):
    bad = {**valid_customer, "tenure_months": "abc"}
    r = await client.post("/predict", json=bad)
    assert r.status_code == 422


@pytest.mark.anyio
async def test_predict_empty_body_returns_422(client):
    r = await client.post("/predict", json={})
    assert r.status_code == 422


@pytest.mark.anyio
async def test_predict_tenure_out_of_range_returns_422(client, valid_customer):
    bad = {**valid_customer, "tenure_months": 9999}
    r = await client.post("/predict", json=bad)
    assert r.status_code == 422


# ── Predict — risk tier logic ─────────────────────────────────────────────────

@pytest.mark.anyio
async def test_high_risk_customer(client):
    """Month-to-month, short tenure, many tickets → likely high risk."""
    r = await client.post("/predict", json={
        "tenure_months": 2,
        "monthly_charges": 95.0,
        "total_charges": 190.0,
        "contract_type": "month-to-month",
        "payment_method": "electronic_check",
        "support_tickets_30d": 6,
    })
    assert r.status_code == 200
    body = r.json()
    # Probability should be elevated; tier should be medium or high
    assert body["churn_probability"] > 0.3


@pytest.mark.anyio
async def test_low_risk_customer(client):
    """Two-year contract, long tenure, no tickets → low risk."""
    r = await client.post("/predict", json={
        "tenure_months": 60,
        "monthly_charges": 45.0,
        "total_charges": 2700.0,
        "contract_type": "two_year",
        "payment_method": "bank_transfer",
        "support_tickets_30d": 0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["churn_probability"] < 0.7  # should be lower than high-risk


# ── Batch predict ─────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_batch_predict_csv(client):
    csv_content = (
        "tenure_months,monthly_charges,total_charges,contract_type,payment_method,support_tickets_30d\n"
        "12,55.0,660.0,month-to-month,electronic_check,3\n"
        "48,40.0,1920.0,two_year,bank_transfer,0\n"
        "5,99.0,495.0,month-to-month,mailed_check,5\n"
    )
    r = await client.post(
        "/predict/batch",
        files={"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["total_rows"] == 3
    assert len(body["predictions"]) == 3


@pytest.mark.anyio
async def test_batch_non_csv_rejected(client):
    r = await client.post(
        "/predict/batch",
        files={"file": ("data.json", io.BytesIO(b'{"a":1}'), "application/json")},
    )
    assert r.status_code == 400


# ── Customers ─────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_at_risk_customers_default_limit(client):
    r = await client.get("/customers/at-risk")
    assert r.status_code == 200
    body = r.json()
    assert "customers" in body
    assert len(body["customers"]) <= 50


@pytest.mark.anyio
async def test_at_risk_customers_custom_limit(client):
    r = await client.get("/customers/at-risk?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert len(body["customers"]) <= 10


@pytest.mark.anyio
async def test_at_risk_sorted_by_probability(client):
    r = await client.get("/customers/at-risk?limit=20")
    assert r.status_code == 200
    probs = [c["churn_probability"] for c in r.json()["customers"]]
    assert probs == sorted(probs, reverse=True), "Customers not sorted by churn probability"


@pytest.mark.anyio
async def test_customer_detail_exists(client):
    # First fetch list, then detail for first customer
    r = await client.get("/customers/at-risk?limit=1")
    assert r.status_code == 200
    customers = r.json()["customers"]
    if customers:
        cid = customers[0]["id"]
        r2 = await client.get(f"/customers/{cid}")
        assert r2.status_code == 200
        assert r2.json()["id"] == cid


@pytest.mark.anyio
async def test_customer_not_found_returns_404(client):
    r = await client.get("/customers/DOES-NOT-EXIST-99999")
    assert r.status_code == 404


# ── Model metadata ────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_model_metadata_returns_metrics(client):
    r = await client.get("/model/metadata")
    assert r.status_code == 200
    body = r.json()
    assert "roc_auc" in body
    assert "model_version" in body
    assert 0.0 < body["roc_auc"] <= 1.0
