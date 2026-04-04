"""Shared constants for Qwen3-8B tool-use fine-tuning pipeline."""

TOOL_USE_SYSTEM_PROMPT = (
    "You are a helpful assistant that can use tools. "
    "When you need to use a tool, respond with a JSON tool call "
    "inside <tool_call> tags, like:\n"
    "<tool_call>\n"
    '{"name": "tool_name", "arguments": {"arg": "value"}}\n'
    "</tool_call>"
)
