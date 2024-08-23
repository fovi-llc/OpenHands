from typing import cast

from agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.controller.state.state import State
from openhands.core.config import load_app_config
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.events.action.action import Action
from openhands.events.action.empty import NullAction
from openhands.events.action.message import MessageAction
from openhands.events.event import Event, EventSource
from openhands.events.observation.commands import (
    CmdOutputObservation,
    IPythonRunCellObservation,
)
from openhands.events.observation.empty import NullObservation
from openhands.events.observation.error import ErrorObservation
from openhands.events.observation.observation import Observation
from openhands.llm.llm import LLM

config = load_app_config()
llm = LLM(config.get_llm_config())


class StuckDetector:
    def __init__(self, state: State):
        self.state = state

    def generate_resolution(self, actions, observations):
        #
        struck_prompt = 'You analyze the history to find out why the agent is stuck and generate a resolution'
        # stuck input
        stuck_input = 'Aanalyze the history'
        for index, (action, observation) in enumerate(zip(actions, observations), 1):
            action = CodeActAgent.get_action_message(action)
            observation = CodeActAgent.get_observation_message(observation)
            stuck_input += f'\n{index = }. {action = }\n{observation = }'

        message_sequence = []
        message_sequence.append(
            Message(role='system', content=[TextContent(text=struck_prompt)])
        )
        message_sequence.append(
            Message(role='user', content=[TextContent(text=stuck_input)])
        )

        response = llm.completion(
            messages=message_sequence,
            temperature=0.0,
            condense=True,
        )
        return response.choices[0].message.content

    def get_last_event_pairs(
        self, history: list[Event], n: int
    ) -> tuple[list[Event], list[Event]]:
        if len(history) < n:
            return [], []

        last_actions: list[Event] = []
        last_observations: list[Event] = []

        for event in reversed(history):
            if isinstance(event, Action) and len(last_actions) < n:
                last_actions.append(event)
            elif isinstance(event, Observation) and len(last_observations) < n:
                last_observations.append(event)

            if len(last_actions) == n and len(last_observations) == n:
                break

        return last_actions, last_observations

    def is_stuck(self) -> tuple[bool, str | None]:
        # filter out MessageAction with source='user' from history
        filtered_history = [
            event
            for event in self.state.history.get_events()
            if not (
                (isinstance(event, MessageAction) and event.source == EventSource.USER)
                or
                # there might be some NullAction or NullObservation in the history at least for now
                isinstance(event, NullAction)
                or isinstance(event, NullObservation)
            )
        ]

        last_actions, last_observations = self.get_last_event_pairs(filtered_history, 3)

        if not last_actions:
            return False, None

        # scenario 1: same action, same observation
        if self._is_stuck_repeating_action_observation(last_actions, last_observations):
            return True, self.generate_resolution(last_actions, last_observations)

        # scenario 2: same action, errors
        if self._is_stuck_repeating_action_error(last_actions, last_observations):
            return True, self.generate_resolution(last_actions, last_observations)

        # scenario 3: monologue
        if self._is_stuck_monologue(filtered_history):
            return True, 'You repeated the same message three times'

        # scenario 4: action, observation pattern on the last six steps
        last_six_actions, last_six_observations = self.get_last_event_pairs(
            filtered_history, 6
        )
        if not last_six_actions:
            return False, None
        if self._is_stuck_action_observation_pattern(
            last_six_actions, last_six_observations
        ):
            # (action_1, obs_1), (action_2, obs_2), (action_1, obs_1), (action_2, obs_2)
            return True, self.generate_resolution(
                last_six_actions, last_six_observations
            )

        return False, None

    def _is_stuck_repeating_action_observation(self, last_actions, last_observations):
        # scenario 1: same action, same observation
        # it takes 4 actions and 4 observations to detect a loop
        # assert len(last_actions) == 4 and len(last_observations) == 4

        # reset almost_stuck reminder
        self.state.almost_stuck = 0

        # almost stuck? if two actions, obs are the same, we're almost stuck
        if len(last_actions) >= 2 and len(last_observations) >= 2:
            actions_equal = all(
                self._eq_no_pid(last_actions[0], action) for action in last_actions[:2]
            )
            observations_equal = all(
                self._eq_no_pid(last_observations[0], observation)
                for observation in last_observations[:2]
            )

            # the last two actions and obs are the same?
            if actions_equal and observations_equal:
                self.state.almost_stuck = 2

            # the last three actions and observations are the same?
            if len(last_actions) >= 3 and len(last_observations) >= 3:
                if (
                    actions_equal
                    and observations_equal
                    and self._eq_no_pid(last_actions[0], last_actions[2])
                    and self._eq_no_pid(last_observations[0], last_observations[2])
                ):
                    self.state.almost_stuck = 1

            if len(last_actions) == 4 and len(last_observations) == 4:
                if (
                    actions_equal
                    and observations_equal
                    and self._eq_no_pid(last_actions[0], last_actions[3])
                    and self._eq_no_pid(last_observations[0], last_observations[3])
                ):
                    logger.warning('Action, Observation loop detected')
                    self.state.almost_stuck = 0
                    return True

        return False

    def _is_stuck_repeating_action_error(self, last_actions, last_observations):
        # scenario 2: same action, errors
        # it takes 4 actions and 4 observations to detect a loop
        # check if the last four actions are the same and result in errors

        # are the last four actions the same?
        if len(last_actions) == 4 and all(
            self._eq_no_pid(last_actions[0], action) for action in last_actions
        ):
            # and the last four observations all errors?
            if all(isinstance(obs, ErrorObservation) for obs in last_observations):
                logger.warning('Action, ErrorObservation loop detected')
                return True
            # or, are the last four observations all IPythonRunCellObservation with SyntaxError?
            elif all(
                isinstance(obs, IPythonRunCellObservation) for obs in last_observations
            ) and all(
                cast(IPythonRunCellObservation, obs)
                .content[-100:]
                .find('SyntaxError: unterminated string literal (detected at line')
                != -1
                and len(
                    cast(IPythonRunCellObservation, obs).content.split(
                        'SyntaxError: unterminated string literal (detected at line'
                    )[-1]
                )
                < 10
                for obs in last_observations
            ):
                logger.warning('Action, IPythonRunCellObservation loop detected')
                return True
        return False

    def _is_stuck_monologue(self, filtered_history):
        # scenario 3: monologue
        # check for repeated MessageActions with source=AGENT
        # see if the agent is engaged in a good old monologue, telling itself the same thing over and over
        agent_message_actions = [
            (i, event)
            for i, event in enumerate(filtered_history)
            if isinstance(event, MessageAction) and event.source == EventSource.AGENT
        ]

        # last three message actions will do for this check
        if len(agent_message_actions) >= 3:
            last_agent_message_actions = agent_message_actions[-3:]

            if all(
                (last_agent_message_actions[0][1] == action[1])
                for action in last_agent_message_actions
            ):
                # check if there are any observations between the repeated MessageActions
                # then it's not yet a loop, maybe it can recover
                start_index = last_agent_message_actions[0][0]
                end_index = last_agent_message_actions[-1][0]

                has_observation_between = False
                for event in filtered_history[start_index + 1 : end_index]:
                    if isinstance(event, Observation):
                        has_observation_between = True
                        break

                if not has_observation_between:
                    logger.warning('Repeated MessageAction with source=AGENT detected')
                    return True
        return False

    def _is_stuck_action_observation_pattern(
        self, last_six_actions, last_six_observations
    ):
        # scenario 4: action, observation pattern on the last six steps
        # check if the agent repeats the same (Action, Observation)
        # every other step in the last six steps

        # this pattern is every other step, like:
        # (action_1, obs_1), (action_2, obs_2), (action_1, obs_1), (action_2, obs_2),...
        if len(last_six_actions) == 6 and len(last_six_observations) == 6:
            actions_equal = (
                # action_0 == action_2 == action_4
                self._eq_no_pid(last_six_actions[0], last_six_actions[2])
                and self._eq_no_pid(last_six_actions[0], last_six_actions[4])
                # action_1 == action_3 == action_5
                and self._eq_no_pid(last_six_actions[1], last_six_actions[3])
                and self._eq_no_pid(last_six_actions[1], last_six_actions[5])
            )
            observations_equal = (
                # obs_0 == obs_2 == obs_4
                self._eq_no_pid(last_six_observations[0], last_six_observations[2])
                and self._eq_no_pid(last_six_observations[0], last_six_observations[4])
                # obs_1 == obs_3 == obs_5
                and self._eq_no_pid(last_six_observations[1], last_six_observations[3])
                and self._eq_no_pid(last_six_observations[1], last_six_observations[5])
            )

            if actions_equal and observations_equal:
                logger.warning('Action, Observation pattern detected')
                return True
        return False

    def _eq_no_pid(self, obj1, obj2):
        if isinstance(obj1, CmdOutputObservation) and isinstance(
            obj2, CmdOutputObservation
        ):
            # for loop detection, ignore command_id, which is the pid
            return obj1.command == obj2.command and obj1.exit_code == obj2.exit_code
        else:
            # this is the default comparison
            return obj1 == obj2
