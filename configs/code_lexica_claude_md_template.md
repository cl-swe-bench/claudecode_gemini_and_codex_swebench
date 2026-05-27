<!-- code-lexica:start -->
## Code Lexica MCP

This repository has Code Lexica MCP integration. Code Lexica returns
project structure, coding conventions, and system architecture
**pre-filtered by the server to the parts relevant to the specific
task you're working on** — so you can short-circuit codebase
exploration and skip dead-end reads.

### Required parameters on every call

- `repoIdentifier`: `{repo_identifier}` — pass verbatim. Short-form
  names won't resolve.
- `commitHash`: `{commit_hash}` — pass verbatim. Pins the response to
  the exact codebase state being worked on; the server resolves
  task-relevant content against that snapshot rather than the latest
  indexed state.
- `taskPrompt`: the **exact, complete** text of the task / issue you
  were given — pass it **verbatim**. Do **not** summarize, paraphrase,
  shorten, or re-word it. This drives the server-side relevance filter,
  which degrades on a lossy summary; without it you'll get an unfocused
  project overview instead of a task-tailored view.

### When to call

**Before searching this codebase or delegating to a subagent (Explore, Agent, etc.)**,
call `mcp__code-lexica__get_codebase_context`. The response tells you
which files / directories are relevant to your task, what they're
named, how modules connect, and the conventions that apply.

**Use the response to direct your subsequent reads and searches —
don't ignore it and restart from scratch.** The whole point of the
call is to spend many fewer turns exploring; if you treat it as
background reading and re-grep the codebase from a blank slate, the
tool call was wasted.

### Subagents

Subagents inherit access to this MCP tool, but you should NOT have
subagents call `get_codebase_context` themselves — that fetches the
same data twice and bloats the conversation cache. Instead: call
`get_codebase_context` ONCE at the top of your work, then INCLUDE the
returned context verbatim in any subagent brief.
<!-- code-lexica:end -->
