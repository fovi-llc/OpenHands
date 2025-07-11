"""
OpenHands MCP Server

This module implements a standalone MCP server that exposes OpenHands functionality
to external MCP clients, allowing other applications to use OpenHands capabilities
through the Model Context Protocol.
"""

import asyncio
import json
import os
from typing import Annotated, Optional

import uvicorn
from fastapi import FastAPI
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from openhands.core.config import OpenHandsConfig
from openhands.core.config.utils import finalize_config
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import run_controller
from openhands.core.setup import create_agent, create_runtime, generate_sid
from openhands.events.action import MessageAction
from openhands.runtime.base import Runtime

# Global runtime and config for the MCP server
_global_runtime: Optional[Runtime] = None
_global_config: Optional[OpenHandsConfig] = None

# Initialize FastMCP server
mcp_server = FastMCP('openhands', dependencies=[])


async def initialize_openhands(config: OpenHandsConfig) -> None:
    """Initialize OpenHands runtime and configuration."""
    global _global_runtime, _global_config
    _global_config = config

    # Create a runtime for executing actions
    _global_runtime = create_runtime(config, sid=generate_sid(config))
    await _global_runtime.connect()


@mcp_server.tool()
async def run_task(
    task: Annotated[
        str, Field(description='The task description for OpenHands to execute')
    ],
    agent_name: Annotated[
        str, Field(description='Name of the agent to use (default: CodeActAgent)')
    ] = 'CodeActAgent',
    max_iterations: Annotated[
        int, Field(description='Maximum number of iterations to run (default: 10)')
    ] = 10,
    workspace_dir: Annotated[
        Optional[str], Field(description='Working directory for the task (optional)')
    ] = None,
) -> str:
    """
    Execute a task using OpenHands with a specified agent.

    Args:
        task: The task description for OpenHands to execute
        agent_name: Name of the agent to use (default: CodeActAgent)
        max_iterations: Maximum number of iterations to run (default: 10)
        workspace_dir: Working directory for the task (optional)

    Returns:
        A summary of the task execution results
    """
    if not _global_config or not _global_runtime:
        raise ToolError('OpenHands is not properly initialized')

    try:
        # Create a copy of the config for this task
        config = _global_config.model_copy()
        config.max_iterations = max_iterations

        # Set workspace directory if provided
        if workspace_dir:
            if not os.path.exists(workspace_dir):
                os.makedirs(workspace_dir, exist_ok=True)
            config.sandbox.workspace_base = workspace_dir

        # Create agent
        agent = create_agent(config, agent_name)

        # Create runtime for this task
        sid = generate_sid(config)
        runtime = create_runtime(config, sid=sid)
        await runtime.connect()

        try:
            # Create initial action
            initial_action = MessageAction(content=task)

            # Run the controller
            final_state = await run_controller(
                config=config,
                initial_user_action=initial_action,
                runtime=runtime,
                agent=agent,
                sid=sid,
                headless_mode=True,
            )

            if final_state:
                # Prepare response with execution summary
                result = {
                    'task': task,
                    'agent_used': agent_name,
                    'iterations_completed': len(final_state.history),
                    'exit_reason': final_state.last_error or 'Completed successfully',
                    'final_agent_state': final_state.agent_state.value
                    if final_state.agent_state
                    else 'unknown',
                }

                # Include the last few events for context
                recent_events = []
                for event in final_state.history[-5:]:  # Last 5 events
                    event_data = {
                        'source': event.source.value
                        if hasattr(event, 'source')
                        else 'unknown',
                        'timestamp': event.timestamp.isoformat()
                        if hasattr(event, 'timestamp')
                        else None,
                    }

                    # Add specific event details based on type
                    if hasattr(event, 'content'):
                        event_data['content'] = event.content[
                            :500
                        ]  # Truncate long content
                    elif hasattr(event, 'observation'):
                        event_data['observation'] = str(event.observation)[:500]

                    recent_events.append(event_data)

                result['recent_events'] = recent_events

                return json.dumps(result, indent=2)
            else:
                return json.dumps(
                    {
                        'error': 'Task execution failed - no final state returned',
                        'task': task,
                        'agent_used': agent_name,
                    },
                    indent=2,
                )

        finally:
            # Clean up runtime
            await runtime.close()

    except Exception as e:
        logger.error(f'Error executing task: {e}', exc_info=True)
        raise ToolError(f'Failed to execute task: {str(e)}')


@mcp_server.tool()
async def execute_command(
    command: Annotated[str, Field(description='Shell command to execute')],
    working_dir: Annotated[
        Optional[str], Field(description='Working directory to execute the command in')
    ] = None,
) -> str:
    """
    Execute a shell command using OpenHands runtime.

    Args:
        command: Shell command to execute
        working_dir: Working directory to execute the command in (optional)

    Returns:
        Command output and exit code
    """
    if not _global_runtime:
        raise ToolError('OpenHands runtime is not initialized')

    try:
        from openhands.events.action import CmdRunAction

        # Create command action
        if working_dir:
            # Change to working directory and execute command
            full_command = f'cd {working_dir} && {command}'
        else:
            full_command = command

        action = CmdRunAction(command=full_command)

        # Execute the command
        observation = _global_runtime.run_action(action)

        result = {
            'command': command,
            'working_dir': working_dir,
            'exit_code': getattr(observation, 'exit_code', None),
            'content': getattr(observation, 'content', str(observation)),
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f'Error executing command: {e}', exc_info=True)
        raise ToolError(f'Failed to execute command: {str(e)}')


@mcp_server.tool()
async def read_file(
    file_path: Annotated[str, Field(description='Path to the file to read')],
    start_line: Annotated[
        Optional[int], Field(description='Starting line number (1-based, optional)')
    ] = None,
    end_line: Annotated[
        Optional[int], Field(description='Ending line number (1-based, optional)')
    ] = None,
) -> str:
    """
    Read a file using OpenHands runtime.

    Args:
        file_path: Path to the file to read
        start_line: Starting line number (1-based, optional)
        end_line: Ending line number (1-based, optional)

    Returns:
        File content or specified line range
    """
    if not _global_runtime:
        raise ToolError('OpenHands runtime is not initialized')

    try:
        from openhands.events.action import FileReadAction

        # Create file read action
        action = FileReadAction(path=file_path)

        # Execute the action
        observation = _global_runtime.run_action(action)

        content = getattr(observation, 'content', str(observation))

        # Handle line range if specified
        if start_line is not None or end_line is not None:
            lines = content.split('\n')
            start_idx = (start_line - 1) if start_line else 0
            end_idx = end_line if end_line else len(lines)
            content = '\n'.join(lines[start_idx:end_idx])

        result = {
            'file_path': file_path,
            'start_line': start_line,
            'end_line': end_line,
            'content': content,
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f'Error reading file: {e}', exc_info=True)
        raise ToolError(f'Failed to read file: {str(e)}')


@mcp_server.tool()
async def write_file(
    file_path: Annotated[str, Field(description='Path to the file to write')],
    content: Annotated[str, Field(description='Content to write to the file')],
    create_dirs: Annotated[
        bool,
        Field(description="Whether to create parent directories if they don't exist"),
    ] = True,
) -> str:
    """
    Write content to a file using OpenHands runtime.

    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        create_dirs: Whether to create parent directories if they don't exist

    Returns:
        Success confirmation with file details
    """
    if not _global_runtime:
        raise ToolError('OpenHands runtime is not initialized')

    try:
        from openhands.events.action import FileWriteAction

        # Create parent directories if requested
        if create_dirs:
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

        # Create file write action
        action = FileWriteAction(path=file_path, content=content)

        # Execute the action
        observation = _global_runtime.run_action(action)

        result = {
            'file_path': file_path,
            'content_length': len(content),
            'success': True,
            'message': getattr(observation, 'content', 'File written successfully'),
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f'Error writing file: {e}', exc_info=True)
        raise ToolError(f'Failed to write file: {str(e)}')


@mcp_server.tool()
async def edit_file(
    file_path: Annotated[str, Field(description='Path to the file to edit')],
    old_str: Annotated[str, Field(description='String to replace in the file')],
    new_str: Annotated[
        str, Field(description='New string to replace the old string with')
    ],
) -> str:
    """
    Edit a file by replacing text using OpenHands runtime.

    Args:
        file_path: Path to the file to edit
        old_str: String to replace in the file
        new_str: New string to replace the old string with

    Returns:
        Edit confirmation with details
    """
    if not _global_runtime:
        raise ToolError('OpenHands runtime is not initialized')

    try:
        from openhands.events.action import FileEditAction

        # Create file edit action
        action = FileEditAction(path=file_path, old_str=old_str, new_str=new_str)

        # Execute the action
        observation = _global_runtime.run_action(action)

        result = {
            'file_path': file_path,
            'old_str': old_str,
            'new_str': new_str,
            'success': True,
            'message': getattr(observation, 'content', 'File edited successfully'),
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f'Error editing file: {e}', exc_info=True)
        raise ToolError(f'Failed to edit file: {str(e)}')


@mcp_server.tool()
async def list_agents() -> str:
    """
    List all available OpenHands agents.

    Returns:
        JSON list of available agent names and descriptions
    """
    try:
        from openhands.agenthub import _ALL_AGENTS

        agents = []
        for agent_name, agent_cls in _ALL_AGENTS.items():
            agent_info = {
                'name': agent_name,
                'class': agent_cls.__name__,
                'module': agent_cls.__module__,
                'description': getattr(
                    agent_cls, '__doc__', 'No description available'
                ),
            }
            agents.append(agent_info)

        result = {
            'available_agents': agents,
            'total_count': len(agents),
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f'Error listing agents: {e}', exc_info=True)
        raise ToolError(f'Failed to list agents: {str(e)}')


@mcp_server.tool()
async def get_server_status() -> str:
    """
    Get the current status of the OpenHands MCP server.

    Returns:
        Server status information including runtime state and configuration
    """
    try:
        runtime_initialized = _global_runtime is not None
        config_loaded = _global_config is not None

        status = {
            'server_status': 'running',
            'runtime_initialized': runtime_initialized,
            'config_loaded': config_loaded,
            'openhands_version': 'latest',  # You could import the actual version
        }

        if config_loaded and _global_config:
            status['sandbox_type'] = _global_config.sandbox.type
            status['default_agent'] = _global_config.default_agent
            status['max_iterations'] = _global_config.max_iterations

        return json.dumps(status, indent=2)

    except Exception as e:
        logger.error(f'Error getting server status: {e}', exc_info=True)
        raise ToolError(f'Failed to get server status: {str(e)}')


async def create_mcp_server_app(config: OpenHandsConfig) -> FastAPI:
    """Create and configure the FastAPI app with MCP server."""
    # Initialize OpenHands
    await initialize_openhands(config)

    # Create FastAPI app and mount MCP server
    app = FastAPI(title='OpenHands MCP Server', version='1.0.0')

    # Mount the MCP server using http_app() method
    mcp_app = mcp_server.http_app()
    app.mount('/mcp', mcp_app)

    @app.get('/health')
    async def health_check():
        """Health check endpoint."""
        return {
            'status': 'healthy',
            'runtime_initialized': _global_runtime is not None,
            'config_loaded': _global_config is not None,
        }

    return app


async def run_mcp_server(
    host: str = '127.0.0.1', port: int = 8000, config: Optional[OpenHandsConfig] = None
):
    """Run the OpenHands MCP server."""
    if config is None:
        # Create a default config
        config = OpenHandsConfig()
        config = finalize_config(config)

    logger.info(f'Starting OpenHands MCP Server on {host}:{port}')

    # Create the FastAPI app
    app = await create_mcp_server_app(config)

    # Run the server
    server_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level='info',
        access_log=True,
    )

    server = uvicorn.Server(server_config)

    try:
        await server.serve()
    except KeyboardInterrupt:
        logger.info('Shutting down OpenHands MCP Server...')
    finally:
        # Clean up global runtime
        if _global_runtime:
            _global_runtime.close()


if __name__ == '__main__':
    # Simple CLI runner for testing
    import argparse

    parser = argparse.ArgumentParser(description='Run OpenHands MCP Server')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')

    args = parser.parse_args()

    asyncio.run(run_mcp_server(host=args.host, port=args.port))
