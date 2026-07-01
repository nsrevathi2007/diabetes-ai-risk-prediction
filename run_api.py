"""Launch the FastAPI backend with Uvicorn."""

from __future__ import annotations

import uvicorn

from api.app import create_app


def main() -> None:
    """Run the API server."""
    app = create_app()
    uvicorn.run(app, host=app.state.settings.host, port=app.state.settings.port)


if __name__ == "__main__":
    main()
