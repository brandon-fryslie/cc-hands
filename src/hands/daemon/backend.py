"""Where a setting meets its login: the backend config.toml names, with the login it reaches its model on, in an
environment that names no setting.

Read by `hands run` before the voice loads, by an edit to the settings before it is restarted on, and by `hands check`,
which loads the voice's pipeline for none of them.
"""

from collections.abc import Mapping

from hands.core.wire import UPSTREAM
from hands.daemon.config import Claude
from hands.sessions import claudecode
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.sessions.wrapper import Unpackaged, packaged
from hands.voice.backends import ClaudeCodeBackend


def backend(claude_code: claudecode.ClaudeCode, llm: Claude, home: Home, environment: Mapping[str, str]) -> ClaudeCodeBackend:
    """The brain on the model the settings name, given the login it reaches its model with, as `claude_code` says it;
    raises Rejected naming what it cannot have, and claudecode.Unreachable when the brain's Claude Code cannot be asked."""
    # [LAW:single-enforcer] where a setting meets its login: the brain's login is checked here, once, before the voice
    # loads, rather than once every turn has failed.
    # [LAW:no-silent-failure] a setting in the environment would be one silently not applied: settings are the home's
    # config.toml, and HANDS_HOME, where that is, is the one variable of hands' own it reads.
    if stray := sorted(name for name in environment if name.startswith("HANDS_") and name != "HANDS_HOME"):
        raise Rejected(f"{', '.join(stray)} set, and hands reads no setting from the environment; settings go in {home.config}")
    # Imported here, so that only a check loads the brain's process, its MCP server, and Pipecat's tools.
    from hands.brain.process import NotLoggedIn, Unstartable, account_kept_out, answered, logged_in

    try:
        account = logged_in(claude_code, home.brain, UPSTREAM, environment)
        answered(claude_code, home.brain)
        account_kept_out(claude_code, home.brain)
        # The brain runs under the fritter hands' package carries: a hands built without it is refused here, not at its start.
        packaged()
    except (NotLoggedIn, Unstartable, Unpackaged) as error:
        raise Rejected(str(error)) from error
    return ClaudeCodeBackend(model=llm.model, config_dir=home.brain, account=account)
