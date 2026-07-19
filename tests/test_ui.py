from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_initial_view_renders_without_error() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    assert not app.exception
    assert app.selectbox[0].value == "All supported antibiotics"
    assert any(button.label == "Run prediction" for button in app.button)
    demo_labels = [button.label for button in app.button if button.label.startswith("Run demo")]
    assert demo_labels == [
        "Run demo 1 · Mixed response",
        "Run demo 2 · Safety no-call",
        "Run demo 3 · Both likely to work",
    ]
    assert len(app.warning) == 1

    demo_button = next(button for button in app.button if button.label == demo_labels[1])
    app = demo_button.click().run(timeout=60)
    assert not app.exception
    assert len(app.tabs) == 2
    assert any(subheader.value == "Prediction report" for subheader in app.subheader)
    assert any("probability in uncertain zone" in info.value for info in app.info)
    assert len(app.warning) == 2
