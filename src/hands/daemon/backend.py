"""Where a setting meets its secret: the backend config.toml names, with the key or the login it reaches its model on.

Read by `hands run` before the voice loads, by an edit to the settings before it is restarted on, and by `hands check`,
which loads the voice's pipeline for none of them.
"""

import atexit
import subprocess
from collections.abc import Mapping

from hands.core.wire import UPSTREAM
from hands.daemon.config import ANTHROPIC_URL, LLM, Anthropic, Claude, OpenAI
from hands.sessions.home import Home
from hands.sessions.payload import Rejected
from hands.voice.backends import AnthropicBackend, ClaudeCodeBackend, LLMBackend, OpenAICompatibleBackend

# Where the Anthropic key lives when ANTHROPIC_API_KEY is not set: a generic password in the keychain.
# A prompt to allow access that nobody answers is a failed read, not a daemon that never starts.
KEYCHAIN_TIMEOUT_SECONDS = 30.0
ANTHROPIC_KEYCHAIN_SERVICE = "HANDS_LLM_ANT_KEY"


def backend(llm: LLM, home: Home, environment: Mapping[str, str]) -> LLMBackend:
    """The backend the settings name, given the key or the login it reaches its model with; raises Rejected naming what it cannot have."""
    # [LAW:single-enforcer] where a setting meets its secret: the key from the environment, or the keychain, or the
    # brain's login, is checked here, once, before the voice loads, rather than once every turn has failed.
    match llm:
        case OpenAI(url=url, model=model):
            return OpenAICompatibleBackend(base_url=url, api_key=_key(environment, "OPENAI_API_KEY"), model=model)
        case Anthropic(url=url, model=model) if url == ANTHROPIC_URL:
            key = _environment_key(environment, "ANTHROPIC_API_KEY") or _keychain_key(ANTHROPIC_KEYCHAIN_SERVICE, "ANTHROPIC_API_KEY")
            return AnthropicBackend(base_url=url, api_key=key, model=model)
        case Anthropic(url=url, model=model):
            # The keychain's key is Anthropic's own, so it is never sent to another server: that server's key is named in the environment.
            return AnthropicBackend(base_url=url, api_key=_key(environment, "ANTHROPIC_API_KEY"), model=model)
        case Claude(model=model):
            # Imported here, so that only a brain's check loads the brain's process, its MCP server, and Pipecat's tools.
            from hands.brain.process import NotLoggedIn, Unstartable, account_kept_out, answered, logged_in

            try:
                account = logged_in(home.brain, UPSTREAM, environment)
                answered(home.brain)
                account_kept_out(home.brain)
            except (NotLoggedIn, Unstartable) as error:
                raise Rejected(str(error)) from error
            return ClaudeCodeBackend(model=model, config_dir=home.brain, account=account)


def _key(environment: Mapping[str, str], var: str) -> str:
    """The API key a keyed backend cannot run without; raises Rejected naming the variable."""
    key = _environment_key(environment, var)
    if not key:
        raise Rejected(f"{var} is not set; the [llm] backend hands is set to run on needs it to reach its model.")
    return key


def _environment_key(environment: Mapping[str, str], var: str) -> str:
    # A key has no whitespace in it: space around one in a .env is dropped, and a blank one is no key.
    return environment.get(var, "").strip()


def _keychain_key(service: str, var: str) -> str:
    """The key the keychain holds under `service`; when it holds none, raises Rejected naming both places a key can be."""
    key = keychain_password(service)
    if not key:
        raise Rejected(f"{var} is not set and the keychain holds no {service}; the [llm] backend hands is set to run on needs one to reach its model.")
    return key


def keychain_password(service: str) -> str | None:
    """The generic password the keychains on the search list hold for `service`, or None when they hold none."""
    bypass = "set ANTHROPIC_API_KEY to start without the keychain"
    with subprocess.Popen(["security", "find-generic-password", "-s", service, "-w"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as found:
        # The read runs on a daemon thread, which a stop mid-prompt exits without: the prompt goes with the process
        # that asked, not left on screen for a daemon that is gone.
        atexit.register(found.kill)
        try:
            out, err = found.communicate(timeout=KEYCHAIN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            found.kill()
            raise Rejected(f"reading {service} from the keychain waited {KEYCHAIN_TIMEOUT_SECONDS:.0f}s, likely on a prompt to allow access; {bypass}.")
        finally:
            atexit.unregister(found.kill)
    # [LAW:no-silent-failure] 44 is `security`'s "not found"; any other failure, a locked keychain or a denied prompt, is not an absence.
    if found.returncode == 44:
        return None
    if found.returncode != 0:
        raise Rejected(f"reading {service} from the keychain failed: {err.strip()}; {bypass}.")
    return out.strip() or None
