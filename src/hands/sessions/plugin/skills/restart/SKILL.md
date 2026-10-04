---
description: Restart hands, so it runs with the code, the brain's prompt, and the brain's setup on disk now. hands stops and starts again in the same terminal and menu-bar item, keeps every running session, and is back once its speech models load. No change is taken up without a restart; a skill added to this plugin is the session's own to load, with /reload-plugins.
allowed-tools: Bash(*hands.daemon.restart*)
---

!`"${CLAUDE_PLUGIN_ROOT}/hooks/python" -m hands.daemon.restart`

Tell the user, in one line, what the line above says. Do nothing else.
