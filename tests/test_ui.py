from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

import app as ui


def test_hosted_launcher_preserves_xsrf_with_iframe_cookie() -> None:
    import hosted_streamlit

    assert hosted_streamlit.server.TORNADO_SETTINGS["xsrf_cookie_kwargs"] == {
        "samesite": "None",
        "secure": True,
    }


def test_streamlit_uploader_has_no_lazy_axios_chunk() -> None:
    assert ui.st.__version__ == "1.36.0"
    static_root = Path(ui.st.__file__).resolve().parent / "static"
    assert not list(static_root.rglob("axios.*.js"))


def test_custom_markup_avoids_lazy_streamlit_html_renderer(monkeypatch) -> None:
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "st.html(" not in source

    rendered: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        ui.st,
        "markdown",
        lambda body, unsafe_allow_html=False: rendered.append((body, unsafe_allow_html)),
    )
    ui._render_html("""
        <section class="test">
          <strong>Rendered</strong>
        </section>
    """)
    assert rendered == [
        ('<section class="test">\n  <strong>Rendered</strong>\n</section>', True)
    ]


def test_hosted_upload_removes_derived_cache(monkeypatch, tmp_path: Path) -> None:
    cache_root = tmp_path / "inference"
    cache_root.mkdir()
    (cache_root / "derived.tsv").write_text("derived", encoding="utf-8")
    monkeypatch.setattr(ui, "EPHEMERAL_INFERENCE", True)
    monkeypatch.setattr(ui, "INFERENCE_CACHE_ROOT", cache_root)
    monkeypatch.setattr(ui, "predict_genome", lambda *args, **kwargs: [{"ok": True}])

    uploaded = SimpleNamespace(name="sample.fna", getvalue=lambda: b">sample\nACGT\n")
    assert ui._run_uploaded(uploaded, None) == [{"ok": True}]
    assert not cache_root.exists()


def test_streamlit_app_initial_view_renders_without_error() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)
    assert not app.exception
    assert any(element.value.startswith("<style>") for element in app.markdown)
    assert any(element.value.startswith('<section class="gf-hero">') for element in app.markdown)
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
