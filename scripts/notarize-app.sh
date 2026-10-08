#!/bin/sh
# notarize-app.sh APP: have Apple notarize the signed hands.app at APP, staple the ticket to it, and check that
# Gatekeeper accepts it as notarized. The notarytool credentials are the keychain profile HANDS_NOTARY_PROFILE names
# (`xcrun notarytool store-credentials` makes one).
set -eu
[ $# -eq 1 ] || { echo "notarize-app: usage: notarize-app.sh APP (given: $*)" >&2; exit 2; }
: "${HANDS_NOTARY_PROFILE:?notarize-app: HANDS_NOTARY_PROFILE names no notarytool keychain profile}"
app=$1
upload=$(mktemp -d)
trap 'rm -rf "$upload"' EXIT
# notarytool takes a zip, not a bundle; ditto keeps the bundle's signature intact in it.
ditto -c -k --keepParent "$app" "$upload/hands.zip"
xcrun notarytool submit "$upload/hands.zip" --keychain-profile "$HANDS_NOTARY_PROFILE" --wait --output-format json > "$upload/result.json" \
  || { cat "$upload/result.json" >&2; echo "notarize-app: notarytool could not submit $app" >&2; exit 1; }
# [LAW:no-silent-failure] notarytool's exit says the submission finished, not that Apple accepted it.
status=$(plutil -extract status raw "$upload/result.json")
[ "$status" = "Accepted" ] || { echo "notarize-app: Apple's verdict was $status; \`xcrun notarytool log $(plutil -extract id raw "$upload/result.json") --keychain-profile $HANDS_NOTARY_PROFILE\` says why" >&2; exit 1; }
xcrun stapler staple "$app"
xcrun stapler validate "$app"
spctl --assess --type execute --verbose=2 "$app" 2>&1 | tee "$upload/assessed"
grep -q "source=Notarized Developer ID" "$upload/assessed" || { echo "notarize-app: Gatekeeper does not take $app as notarized" >&2; exit 1; }
