package main

import (
	"regexp"
	"strings"
)

// opensList reports whether a cursor at the end of text would have a completion list open.
//
// [LAW:one-source-of-truth] The child's own patterns, read out of 2.1.278 rather than
// guessed, and joined into one:
//
//	@ /(^|[\s\u3002\u3001\uFF1F\uFF01])@([\p{L}\p{N}\p{M}_\-./\\()[\]~:]*|"[^"]*"?)$/u
//	# /(^|\s)#([a-z0-9][a-z0-9_-]*)$/
//	: /(^|\s):([a-z0-9_+-]{2,})$/
//
// The `*` on the first is why `@` needs nothing after it: the cursor sitting straight
// after an `@` already opens the list on every file there is.
//
// A slash command is not here. Its Return runs the command and empties the box - the
// child passes `shouldExecute` true on that path - so it is an ordinary submit.
func opensList(text string) bool {
	return listToken.MatchString(text)
}

var listToken = regexp.MustCompile(`(?:^|[\s\x{3002}\x{3001}\x{FF1F}\x{FF01}])@(?:[\p{L}\p{N}\p{M}_\-./\\()\[\]~:]*|"[^"]*"?)$` +
	`|(?:^|\s)#[a-z0-9][a-z0-9_-]*$` +
	`|(?:^|\s):[a-z0-9_+-]{2,}$`)

// staysUnsent reports whether an Enter pressed straight after text, with the cursor at its
// end, would go on with the line instead of sending it: after a backslash, which the child
// turns into a newline, or under a completion list, which takes the Enter for itself.
//
// Text fritter types into a box it has emptied leaves the cursor at its end, so the two
// rules can be asked exactly.
func staysUnsent(text string) bool {
	return strings.HasSuffix(text, `\`) || opensList(text)
}
