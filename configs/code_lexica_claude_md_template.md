<!-- code-lexica:start -->
## Code Lexica MCP

This repository has Code Lexica MCP integration. Code Lexica provides
pre-indexed codebase intelligence — architecture, code maps, coding
conventions, and implementation patterns — that narrows the search
space and shortens exploration loops.

**Repo identifier for this codebase:** `{repo_identifier}`
Pass this verbatim as the `repoIdentifier` parameter on every Code
Lexica call. Short-form names won't resolve.

### When to call

**Before searching this codebase or delegating to a subagent (Explore, Agent, etc.)**,
call `mcp__code-lexica__get_codebase_context` to get the architecture,
code map, and conventions. Then drill in with grep / find / Read using
the narrowed surface — context first means fewer dead-end reads.

### Subagents

Subagents inherit access to this MCP tool, but you should NOT have
subagents call `get_codebase_context` themselves — that fetches the
same data twice and bloats the conversation cache. Instead: call
`get_codebase_context` ONCE at the top of your work, then INCLUDE the
returned context verbatim in any subagent brief.
<!-- code-lexica:end -->
