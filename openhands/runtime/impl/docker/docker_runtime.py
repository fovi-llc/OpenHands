import atexit
import os
import sys
import time
from functools import lru_cache
from typing import Callable

import docker
import requests
import tenacity

from openhands.core.config import AppConfig
from openhands.core.exceptions import (
    AgentRuntimeDisconnectedError,
    AgentRuntimeNotFoundError,
)
from openhands.core.logger import DEBUG, DEBUG_RUNTIME
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.runtime.builder import DockerRuntimeBuilder
from openhands.runtime.impl.action_execution.action_execution_client import (
    ActionExecutionClient,
)
from openhands.runtime.impl.docker.containers import remove_all_containers
from openhands.runtime.plugins import PluginRequirement
from openhands.runtime.utils import find_available_tcp_port
from openhands.runtime.utils.command import get_action_execution_server_startup_command
from openhands.runtime.utils.log_streamer import LogStreamer
from openhands.runtime.utils.runtime_build import build_runtime_image
from openhands.utils.async_utils import call_sync_from_async
from openhands.utils.tenacity_stop import stop_if_should_exit

CONTAINER_NAME_PREFIX = 'kevin-runtime-'

EXECUTION_SERVER_PORT_RANGE = (30000, 39999)
VSCODE_PORT_RANGE = (40000, 49999)
APP_PORT_RANGE_1 = (50000, 54999)
APP_PORT_RANGE_2 = (55000, 59999)


def remove_all_runtime_containers():
    remove_all_containers(CONTAINER_NAME_PREFIX)


_atexit_registered = False


class DockerRuntime(ActionExecutionClient):
    """This runtime will subscribe the event stream.
    When receive an event, it will send the event to runtime-client which run inside the docker environment.

    Args:
        config (AppConfig): The application configuration.
        event_stream (EventStream): The event stream to subscribe to.
        sid (str, optional): The session ID. Defaults to 'default'.
        plugins (list[PluginRequirement] | None, optional): List of plugin requirements. Defaults to None.
        env_vars (dict[str, str] | None, optional): Environment variables to set. Defaults to None.
    """

    def __init__(
        self,
        config: AppConfig,
        event_stream: EventStream,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_callback: Callable | None = None,
        attach_to_existing: bool = False,
        headless_mode: bool = True,
    ):
        global _atexit_registered
        if not _atexit_registered:
            _atexit_registered = True
            atexit.register(remove_all_runtime_containers)

        self.config = config
        self.persist_sandbox = self.config.sandbox.persist_sandbox
        if self.persist_sandbox:
            # odd port number will be used for vscode
            if sys.argv[1:] and 'resolve_issue' in sys.argv[1]:
                self._container_port = 63708
            elif self.config.run_as_openhands:
                user = 'oh'
                self._container_port = self.config.sandbox.port
            else:
                user = 'root'
                self._container_port = 63712
            path = config.workspace_mount_path or sid
            os.environ['selection_id'] = path
            path = ''.join(c if c.isalnum() else '_' for c in path)  # type: ignore
            self.instance_id = f'persisted-{user}-{path}'
        else:
            self.instance_id = sid
            self._container_port = find_available_tcp_port()
        self.container_name = CONTAINER_NAME_PREFIX + self.instance_id
        self.docker_client: docker.DockerClient = self._init_docker_client()
        if not attach_to_existing:
            try:
                self.docker_client.containers.get(self.container_name)
                attach_to_existing = True
            except docker.errors.NotFound:
                attach_to_existing = False
        self._vscode_url: str | None = None  # initial dummy value
        self._runtime_initialized: bool = False
        self.status_callback = status_callback
        self._host_port = self._container_port
        self._vscode_port = -1
        self._app_ports: list[int] = []


        self.base_container_image = self.config.sandbox.base_container_image
        self.runtime_container_image = self.config.sandbox.runtime_container_image
        self.container = None

        self.runtime_builder = DockerRuntimeBuilder(self.docker_client)

        # Buffer for container logs
        self.log_streamer: LogStreamer | None = None

        super().__init__(
            config,
            event_stream,
            sid,
            plugins,
            env_vars,
            status_callback,
            attach_to_existing,
            headless_mode,
        )

        # Log runtime_extra_deps after base class initialization so self.sid is available
        if self.config.sandbox.runtime_extra_deps:
            self.log(
                'debug',
                f'Installing extra user-provided dependencies in the runtime image: {self.config.sandbox.runtime_extra_deps}',
            )

    def _get_action_execution_server_host(self):
        return self.api_url

    async def connect(self):
        self.send_status_message('STATUS$STARTING_RUNTIME')
        if not self.attach_to_existing:
            logger.info('Creating new Docker container')
            if self.runtime_container_image is None:
                if self.base_container_image is None:
                    raise ValueError(
                        'Neither runtime container image nor base container image is set'
                    )
                self.send_status_message('STATUS$STARTING_CONTAINER')
                self.runtime_container_image = build_runtime_image(
                    self.base_container_image,
                    self.runtime_builder,
                    platform=self.config.sandbox.platform,
                    extra_deps=self.config.sandbox.runtime_extra_deps,
                    force_rebuild=self.config.sandbox.force_rebuild_runtime,
                    extra_build_args=self.config.sandbox.runtime_extra_build_args,
                )

            self.log(
                'info', f'Starting runtime with image: {self.runtime_container_image}'
            )
            await call_sync_from_async(self._init_container)
            self.log(
                'info',
                f'Container started: {self.container_name}. VSCode URL: {self.vscode_url}',
            )

        else:
            logger.info(f'Using existing Docker container: {self.container_name}')
            try:
                await call_sync_from_async(self._attach_to_container)
            except docker.errors.NotFound as e:
                if self.attach_to_existing:
                    self.log(
                        'error',
                        f'Container {self.container_name} not found.',
                    )
                    raise e
            self.start_docker_container()
        if DEBUG_RUNTIME:
            self.log_streamer = LogStreamer(self.container, self.log)
        else:
            self.log_streamer = None

        if not self.attach_to_existing:
            self.log('info', f'Waiting for client to become ready at {self.api_url}...')
            self.send_status_message('STATUS$WAITING_FOR_CLIENT')

        try:
            await call_sync_from_async(self._wait_until_alive)
        except Exception as e:
            self.log('error', f'Error waiting for client to become ready: {e}')
            extra_msg = ''
            if self.config.sandbox.use_host_network:
                extra_msg = 'Make sure you have enabled host networking in the docker desktop settings. Steps: https://docs.docker.com/network/drivers/host/#docker-desktop'
            raise Exception(
                f'Unable to connect to the runtime docker container. Please check if the container is running and there is no error in the container logs. {extra_msg}'
            )

        if not self.attach_to_existing:
            self.log('info', 'Runtime is ready.')

        if not self.attach_to_existing:
            await call_sync_from_async(self.setup_initial_env)

        self.log(
            'debug',
            f'Container initialized with plugins: {[plugin.name for plugin in self.plugins]}. VSCode URL: {self.vscode_url}',
        )
        if not self.attach_to_existing:
            self.send_status_message(' ')
        self._runtime_initialized = True

        try:
            path = 'openhands/sel/selenium_session_details.py'
            self.copy_to(path, '/openhands/code/openhands/sel/')
            path = 'openhands/sel/selenium_tester.py'
            self.copy_to(path, '/openhands/code/openhands/sel/')
            logger.info('Copied selenium files to runtime')
        except Exception as e:
            logger.error(f'Error copying selenium files to runtime: {e}')
        try:
            self.copy_to('sandbox.env', '/openhands/code')
        except Exception as e:
            logger.error(f'Error copying sandbox.env to runtime: {e}')

    @staticmethod
    @lru_cache(maxsize=1)
    def _init_docker_client() -> docker.DockerClient:
        try:
            return docker.from_env()
        except Exception as e:
            logger.error(
                f'Is Docker Desktop running?\n{e}',
            )
            os._exit(0)

    def _init_container(self):
        self.log('debug', 'Preparing to start container...')
        self.send_status_message('STATUS$PREPARING_CONTAINER')
        self._vscode_port = self._find_available_port(VSCODE_PORT_RANGE)
        self._app_ports = [
            self._find_available_port(APP_PORT_RANGE_1),
            self._find_available_port(APP_PORT_RANGE_2),
        ]
        self.api_url = f'{self.config.sandbox.local_runtime_url}:{self._container_port}'
        use_host_network = self.config.sandbox.use_host_network
        network_mode: str | None = 'host' if use_host_network else None

        # Initialize port mappings
        port_mapping: dict[str, list[dict[str, str]]] | None = None
        if not use_host_network:
            port_mapping = {
                f'{self._container_port}/tcp': [{'HostPort': str(self._host_port)}],
            }

            if self.vscode_enabled:
                port_mapping[f'{self._vscode_port}/tcp'] = [
                    {'HostPort': str(self._vscode_port)}
                ]

            for port in self._app_ports:
                port_mapping[f'{port}/tcp'] = [{'HostPort': str(port)}]
        else:
            self.log(
                'warn',
                'Using host network mode. If you are using MacOS, please make sure you have the latest version of Docker Desktop and enabled host network feature: https://docs.docker.com/network/drivers/host/#docker-desktop',
            )

        # Combine environment variables
        environment = {
            'port': str(self._container_port),
            'PYTHONUNBUFFERED': 1,
            'VSCODE_PORT': str(self._vscode_port),
        }
        if self.config.debug or DEBUG:
            environment['DEBUG'] = 'true'
        # also update with runtime_startup_env_vars
        environment.update(self.config.sandbox.runtime_startup_env_vars)

        self.log('debug', f'Workspace Base: {self.config.workspace_base}')
        if (
            self.config.workspace_mount_path is not None
            and self.config.workspace_mount_path_in_sandbox is not None
        ):
            # e.g. result would be: {"/home/user/openhands/workspace": {'bind': "/workspace", 'mode': 'rw'}}
            volumes = {
                self.config.workspace_mount_path: {
                    'bind': self.config.workspace_mount_path_in_sandbox,
                    'mode': 'rw',
                }
            }
            # mount sock
            volumes['/var/run/docker.sock'] = {
                'bind': '/var/run/docker.sock',
                'mode': 'rw',
            }
            logger.debug(f'Mount dir: {self.config.workspace_mount_path}')
        else:
            logger.debug(
                'Mount dir is not set, will not mount the workspace directory to the container'
            )
            volumes = None
        self.log(
            'debug',
            f'Sandbox workspace: {self.config.workspace_mount_path_in_sandbox}',
        )

        command = get_action_execution_server_startup_command(
            server_port=self._container_port,
            plugins=self.plugins,
            app_config=self.config,
            use_nice_for_root=False,
        )

        try:
            self.container = self.docker_client.containers.run(
                self.runtime_container_image,
                command=command,
                network_mode=network_mode,
                ports=port_mapping,
                working_dir='/openhands/code/',  # do not change this!
                name=self.container_name,
                detach=True,
                environment=environment,
                volumes=volumes,
                device_requests=(
                    [docker.types.DeviceRequest(capabilities=[['gpu']], count=-1)]
                    if self.config.sandbox.enable_gpu
                    else None
                ),
                **(self.config.sandbox.docker_runtime_kwargs or {}),
            )
            self.log('debug', f'Container started. Server url: {self.api_url}')
            self.send_status_message('STATUS$CONTAINER_STARTED')
        except docker.errors.APIError as e:
            if '409' in str(e):
                self.log(
                    'warning',
                    f'Container {self.container_name} already exists. Removing...',
                )
                remove_all_containers(self.container_name)
                return self._init_container()

            else:
                self.log(
                    'error',
                    f'Error: Instance {self.container_name} FAILED to start container!\n',
                )
                self.log('error', str(e))
                raise e
        except Exception as e:
            self.log(
                'error',
                f'Error: Instance {self.container_name} FAILED to start container!\n',
            )
            self.log('error', str(e))
            self.close()
            raise e

    def _attach_to_container(self):
        self.container = self.docker_client.containers.get(self.container_name)
        for port in self.container.attrs['NetworkSettings']['Ports']:  # type: ignore
            port = int(port.split('/')[0])
            if (
                port >= EXECUTION_SERVER_PORT_RANGE[0]
                and port <= EXECUTION_SERVER_PORT_RANGE[1]
            ):
                self._container_port = port
            if port >= VSCODE_PORT_RANGE[0] and port <= VSCODE_PORT_RANGE[1]:
                self._vscode_port = port
            elif port >= APP_PORT_RANGE_1[0] and port <= APP_PORT_RANGE_1[1]:
                self._app_ports.append(port)
            elif port >= APP_PORT_RANGE_2[0] and port <= APP_PORT_RANGE_2[1]:
                self._app_ports.append(port)
        else:
            # fallback if host network is used
            self._container_port = int(self.container.attrs['Args'][9])  # type: ignore
        self._host_port = self._container_port
        self.api_url = f'{self.config.sandbox.local_runtime_url}:{self._container_port}'
        self.log(
            'debug',
            f'attached to container: {self.container_name} {self._container_port} {self.api_url}',
        )

    @tenacity.retry(
        stop=tenacity.stop_after_delay(120) | stop_if_should_exit(),
        retry=tenacity.retry_if_exception_type(
            (ConnectionError, requests.exceptions.ConnectionError)
        ),
        reraise=True,
        wait=tenacity.wait_fixed(2),
    )
    def _wait_until_alive(self):
        try:
            container = self.docker_client.containers.get(self.container_name)
            if container.status == 'exited':
                raise AgentRuntimeDisconnectedError(
                    f'Container {self.container_name} has exited.'
                )
        except docker.errors.NotFound:
            raise AgentRuntimeNotFoundError(
                f'Container {self.container_name} not found.'
            )

        self.check_if_alive()

    def start_docker_container(self):
        try:
            container = self.docker_client.containers.get(self.container_name)
            logger.info('Container status: %s', container.status)
            if container.status != 'running':
                container.start()
                logger.info('Container started')
                self.setup_initial_env(force=True)
            elapsed = 0
            while container.status != 'running':
                time.sleep(1)
                elapsed += 1
                if elapsed > self.config.sandbox.timeout:
                    break
                container = self.docker_client.containers.get(self.container_name)
        except Exception:
            logger.exception('Failed to start container')

    def close(self, rm_all_containers: bool | None = None):
        """Closes the DockerRuntime and associated objects

        Parameters:
        - rm_all_containers (bool): Whether to remove all containers with the 'openhands-sandbox-' prefix
        """
        super().close()
        if self.log_streamer:
            self.log_streamer.close()

        if rm_all_containers is None:
            rm_all_containers = self.config.sandbox.rm_all_containers

        if self.config.sandbox.keep_runtime_alive or self.attach_to_existing:
            return
        close_prefix = (
            CONTAINER_NAME_PREFIX if rm_all_containers else self.container_name
        )
        remove_all_containers(close_prefix)

    def _is_port_in_use_docker(self, port):
        containers = self.docker_client.containers.list()
        for container in containers:
            container_ports = container.ports
            if str(port) in str(container_ports):
                return True
        return False

    def _find_available_port(self, port_range, max_attempts=5):
        port = port_range[1]
        for _ in range(max_attempts):
            port = find_available_tcp_port(port_range[0], port_range[1])
            if not self._is_port_in_use_docker(port):
                return port
        # If no port is found after max_attempts, return the last tried port
        return port

    @property
    def vscode_url(self) -> str | None:
        token = super().get_vscode_token()
        if not token:
            return None

        vscode_url = f'http://localhost:{self._vscode_port}/?tkn={token}&folder={self.config.workspace_mount_path_in_sandbox}'
        return vscode_url

    @property
    def web_hosts(self):
        hosts: dict[str, int] = {}

        for port in self._app_ports:
            hosts[f'http://localhost:{port}'] = port

        return hosts
