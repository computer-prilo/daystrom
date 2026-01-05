# daystrom
Daystrom is an agent framework for easily building workflows. Give AI just enough power, but not too much. The M4 was better than the M5.

This project is still under active initial development and should not be exepected to be stable at this time.

## Quickstart

Daystrom provides convenient primitives for building AI powered workflows. The main primitives you'll want to use directly are:

| primitive  | description |
| ---------- | ----------- |
| Agent      | LLM with access to tools in a loop |
| LLM        | Base component for direct LLM interaction |
| Instructor | Structured Output component leveraging the [Instructor](https://github.com/567-labs/instructor) library |
| @tool      | decorator that wraps any function and makes it a tool |
