#!/bin/bash
# PreToolUse hook: block writes to protected files.

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty')
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Protected files (relative to project dir, matched against basename or full path)
PROTECTED=(
"evaluate.py"
"tinyphysics.py"
"models/"
"data/"
"pyproject.toml"
"program.md"
)

deny() {
jq -n --arg reason "$1" '{
    hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: $reason
    }
}'
exit 0
}

[ -z "$FILE_PATH" ] && exit 0

BASENAME=$(basename "$FILE_PATH")

for protected in "${PROTECTED[@]}"; do
if [[ "$BASENAME" == "$protected" ]]; then
    deny "Write blocked: You can't modify $FILE_PATH"
fi
done

exit 0
