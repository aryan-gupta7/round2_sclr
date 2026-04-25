from openenv.core.env_server.http_server import create_app

try:
    from ..models import RecallAction, RecallObservation
    from .recall_env_environment import RecallEnvironment
except (ImportError, ModuleNotFoundError):
    from models import RecallAction, RecallObservation
    from recall_env_environment import RecallEnvironment

# Create the app with web interface
app = create_app(
    RecallEnvironment,
    RecallAction,
    RecallObservation,
    env_name="recall",
    max_concurrent_envs=1  # We don't support concurrent sessions for now
)

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main()
