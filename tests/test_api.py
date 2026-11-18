
import sys
from pathlib import Path

# Locate project directory
PROJECT_ROOT = Path(__file__).resolve().parent

if not (PROJECT_ROOT / "api.py").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

# Add project directory to Python path
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_valid_prediction():
    response = client.post(
        "/predict",
        json={
            "age": 35,
            "estimated_salary": 60000
        }
    )

    assert response.status_code == 200
    assert response.json()["prediction"] in [0, 1]


def test_invalid_age():
    response = client.post(
        "/predict",
        json={
            "age": 150,
            "estimated_salary": 60000
        }
    )

    assert response.status_code == 422


def test_negative_salary():
    response = client.post(
        "/predict",
        json={
            "age": 35,
            "estimated_salary": -5000
        }
    )

    assert response.status_code == 422
