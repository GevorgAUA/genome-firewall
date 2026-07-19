"""Launch Streamlit with iframe-compatible, protected XSRF cookies."""

# ruff: noqa: E402, I001

from __future__ import annotations

from streamlit.web.server import server

# Hugging Face embeds the app cross-site. Preserve XSRF validation while allowing
# the browser to send its XSRF cookie from the HTTPS iframe.
server.TORNADO_SETTINGS["xsrf_cookie_kwargs"] = {
    "samesite": "None",
    "secure": True,
}

from streamlit.web.cli import main


if __name__ == "__main__":
    main()
