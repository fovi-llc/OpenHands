enum ActionType {
  // Initializes the agent. Only sent by client.
  INIT = "initialize",

  // Represents a message from the user or agent.
  MESSAGE = "message",

  // Regenerates the last message from the agent.
  REGENERATE = "regenerate",

  // Reads the contents of a file.
  READ = "read",

  // Writes the contents to a file.
  WRITE = "write",

  // Runs a command.
  RUN = "run",

  // Runs a IPython command.
  RUN_IPYTHON = "run_ipython",

  // Opens a web page.
  BROWSE = "browse",

  // Interact with the browser instance.
  BROWSE_INTERACTIVE = "browse_interactive",

  // Delegate a (sub)task to another agent.
  DELEGATE = "delegate",

  // If you're absolutely certain that you've completed your task and have tested your work,
  // use the finish action to stop working.
  FINISH = "finish",

  // Reject a request from user or another agent.
  REJECT = "reject",

  // Adds a task to the plan.
  ADD_TASK = "add_task",

  // Updates a task in the plan.
  MODIFY_TASK = "modify_task",

  // Changes the state of the agent, e.g. to paused or running
  CHANGE_AGENT_STATE = "change_agent_state",
}

export default ActionType;
