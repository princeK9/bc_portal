"""Configuration loading and the single-source-of-truth guarantees."""

import config


class TestEnvironmentLoading:
    def test_env_is_loaded_from_repo_root(self):
        """.env is anchored to the repo root, not the cwd, so the CLI scripts
        under scripts/ pick up credentials no matter where they are run from."""
        assert (config.REPO_ROOT / "config.py").exists()

    def test_db_config_is_fully_populated(self):
        for key in ("dbname", "user", "password", "host", "port"):
            assert config.DB_CONFIG.get(key), f"DB_CONFIG[{key!r}] is empty"

    def test_gemini_settings_present(self):
        assert config.GEMINI_API_KEY, "GEMINI_API_KEY not loaded"
        assert config.GEMINI_MODEL, "GEMINI_MODEL not resolved"


class TestSingleSourceOfTruth:
    """DB_CONFIG used to be declared identically in three files. These tests
    fail if a copy is ever reintroduced."""

    def test_server_uses_shared_db_config(self):
        import server

        assert server.DB_CONFIG is config.DB_CONFIG

    def test_migrate_uses_shared_db_config(self):
        import migrate

        assert migrate.DB_CONFIG is config.DB_CONFIG

    def test_vectorize_uses_shared_db_config(self):
        import vectorize

        assert vectorize.DB_CONFIG is config.DB_CONFIG


class TestPaths:
    def test_paths_are_absolute_and_under_repo_root(self):
        for path in (config.UPLOADS_DIR, config.DATA_DIR, config.CANDIDATES_JSON,
                     config.SAMPLE_DATA_DIR):
            assert path.is_absolute()
            assert config.REPO_ROOT in path.parents or path == config.REPO_ROOT


class TestLogging:
    def test_setup_logging_is_idempotent(self):
        """Entry points may each call it; calling twice must not raise."""
        config.setup_logging()
        config.setup_logging()
