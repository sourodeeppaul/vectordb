"""
Command-line entry point for VectorDB server.

Usage:
    python -m vectordb.server [OPTIONS]
    
Options:
    --host TEXT         Host to bind to (default: 0.0.0.0)
    --port INTEGER      Port to bind to (default: 8000)
    --data-dir TEXT     Data directory path
    --reload            Enable auto-reload
    --workers INTEGER   Number of workers
    --log-level TEXT    Log level (DEBUG, INFO, WARNING, ERROR)
"""

import argparse
import uvicorn

from .config import ServerConfig, set_config
from .app import create_app


def main():
    parser = argparse.ArgumentParser(
        description="VectorDB Server - Vector Database REST API"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./vectordb_data",
        help="Data directory path"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (optional)"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = ServerConfig(
        host=args.host,
        port=args.port,
        data_dir=args.data_dir,
        reload=args.reload,
        workers=args.workers,
        log_level=args.log_level,
        api_key=args.api_key,
    )
    
    set_config(config)
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                    VectorDB Server                       ║
╠══════════════════════════════════════════════════════════╣
║  Host:      {config.host:<44} ║
║  Port:      {config.port:<44} ║
║  Data Dir:  {config.data_dir:<44} ║
║  Log Level: {config.log_level:<44} ║
║  API Docs:  http://{config.host}:{config.port}/docs{' ':<24} ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "vectordb.server.app:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        workers=config.workers,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    main()