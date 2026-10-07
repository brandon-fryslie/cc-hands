"""A tool the model can call: the schema it sees, read once from a plain async body's signature and docstring, and the
body that answers a call. [LAW:decomposition] it knows nothing of who calls it: hands' MCP server is the adapter over
tools, and none of its dependencies are this module's.
"""

import functools
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast, get_args, get_origin, get_type_hints, is_typeddict

import docstring_parser


# What the model is handed back from a call: an object, as every tool API carries a result.
Result = Mapping[str, object]
Body = Callable[..., Awaitable[Result]]
JsonSchema = Mapping[str, object]


@dataclass(frozen=True)
class Tool:
    """One tool: the schema the model sees, the body that answers a call, and what a call means to the conversation."""

    name: str
    description: str
    properties: Mapping[str, JsonSchema]
    required: tuple[str, ...]
    body: Body
    # "reply": the model is asked to go on once it has the result. "silence": the call is the whole reply, unless it
    # hands back an error, which the model is asked to answer: a refused call did nothing, and only the model can retry it.
    # A result {"says": text} is said by hands, as written, whichever the model does.
    then: Literal["reply", "silence"]
    # True when a barge-in must not stop a call part way: its effect would land without its readback heard.
    completes: bool

    @property
    def input_schema(self) -> JsonSchema:
        return {"type": "object", "properties": dict(self.properties), "required": list(self.required)}


def tool(body: Body, *, then: Literal["reply", "silence"] = "reply", completes: bool = False) -> Tool:
    """The body as a tool, its schema read once from its signature and docstring: the name, the text, and each argument's type and line."""
    # [LAW:parse-dont-validate] a signature with a type no schema says is refused here, as the daemon builds its tools.
    docstring = docstring_parser.parse(inspect.getdoc(body) or "")
    lines = {param.arg_name: param.description or "" for param in docstring.params}
    hints = get_type_hints(body)
    parameters = inspect.signature(body).parameters.values()
    properties = {parameter.name: {**_schema(hints[parameter.name]), "description": lines.get(parameter.name, "")} for parameter in parameters}
    required = tuple(parameter.name for parameter in parameters if parameter.default is inspect.Parameter.empty)
    return Tool(body.__name__, (docstring.description or "").strip(), properties, required, _closed(body, properties), then, completes)


def _closed(body: Body, properties: Mapping[str, JsonSchema]) -> Body:
    """The body, refusing a call whose arguments do not fit its signature or the schema the model was shown for them,
    so the model is told, as a result, and can call again."""
    # [LAW:single-enforcer] every adapter calls the tool's body, so the schema it advertises is held here, once.
    # [LAW:one-source-of-truth] held against the schema itself, so what is advertised and what is taken cannot differ.
    signature = inspect.signature(body)

    @functools.wraps(body)
    async def call(**arguments: object) -> Result:
        # A null is an argument left out, which some models send for one they leave empty: it takes the argument's
        # default, and is refused where there is none.
        sent = {name: value for name, value in arguments.items() if value is not None}
        try:
            signature.bind(**sent)
        except TypeError as error:
            return {"error": f"{body.__name__} was called with the wrong arguments: {error}"}
        refused = [misfit for name, value in sent.items() for misfit in _misfits(properties[name], value, name)]
        return {"error": "; ".join(refused)} if refused else await body(**sent)

    return call


def _misfits(schema: JsonSchema, value: object, what: str) -> list[str]:
    """Each way a value the model sent does not fit a schema `_schema` wrote, in words the model can correct it from.

    A value of the wrong type is named by its type, never echoed: it may be a whole draft.
    """
    got = type(value).__name__
    match schema:
        case {"enum": enum} if isinstance(value, str):
            allowed = cast(list[str], enum)
            return [] if value in allowed else [f"{value!r} is no {what}; it is one of {', '.join(allowed)}"]
        case {"type": "string"}:
            return [] if isinstance(value, str) else [f"{what} should be a string, got {got}"]
        case {"type": "boolean"}:
            return [] if isinstance(value, bool) else [f"{what} should be true or false, got {got}"]
        case {"type": "integer"}:
            # A bool is an int to Python, and no integer to the model.
            return [] if isinstance(value, int) and not isinstance(value, bool) else [f"{what} should be an integer, got {got}"]
        case {"type": "array", "items": items}:
            if not isinstance(value, list):
                return [f"{what} should be a list, got {got}"]
            return [misfit for index, item in enumerate(cast(list[object], value)) for misfit in _misfits(cast(JsonSchema, items), item, f"{what}[{index}]")]
        case {"type": "object", "properties": fields, "required": required}:
            if not isinstance(value, dict):
                return [f"{what} should be an object, got {got}"]
            given = cast(dict[str, object], value)
            return [
                *(f"{what} has no {name}" for name in cast(list[str], required) if name not in given),
                *(misfit for name, field in cast(Mapping[str, JsonSchema], fields).items() if name in given for misfit in _misfits(field, given[name], f"{what}.{name}")),
            ]
        case _:
            raise TypeError(f"a tool argument's schema {schema!r} is none written here")


def _schema(hint: object) -> JsonSchema:
    match hint:
        case type() if hint is str:
            return {"type": "string"}
        case type() if hint is bool:
            return {"type": "boolean"}
        case type() if hint is int:
            return {"type": "integer"}
        case type() if is_typeddict(hint):
            fields = get_type_hints(hint)
            return {"type": "object", "properties": {name: _schema(field) for name, field in fields.items()}, "required": list(fields)}
        case _ if get_origin(hint) is Literal:
            return {"type": "string", "enum": list(get_args(hint))}
        case _ if get_origin(hint) is list:
            [item] = get_args(hint)
            return {"type": "array", "items": _schema(item)}
        case _:
            raise TypeError(f"a tool argument typed {hint!r} has no schema here")


def whole(silences: Sequence[bool]) -> bool:
    """Whether a reply's calls were the whole of it: every one of them silent. One that replied or was refused is the model's to answer."""
    return bool(silences) and all(silences)


def silent(tool: Tool, result: Result) -> bool:
    """Whether the call is the whole reply: a silence tool's, unless it was refused."""
    return tool.then == "silence" and "error" not in result
