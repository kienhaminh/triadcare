"""Pydantic models of tool arguments and the JSON schemas the model sees.

The chat template renders tool schemas into the model input, so ToolSchema keeps
them byte-identical to the hand-written schemas of earlier evaluation runs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from pydantic.json_schema import GenerateJsonSchema


class ToolSchema(GenerateJsonSchema):
    """JSON schema without titles or sorting, in the key order of the original schemas.

    An optional field is omitted by the caller, so it shows only its value type.
    """

    def field_title_should_be_set(self, schema) -> bool:
        return False

    def sort(self, value, parent_key=None):
        return value

    def nullable_schema(self, schema):
        return self.generate_inner(schema["schema"])

    def default_schema(self, schema):
        return self.generate_inner(schema["schema"])

    def literal_schema(self, schema):
        generated = super().literal_schema(schema)
        return {"type": generated.pop("type"), **generated}

    def model_schema(self, schema):
        generated = super().model_schema(schema)
        order = ("type", "properties", "required", "additionalProperties")
        return {key: generated[key] for key in order if key in generated}


class ToolArguments(BaseModel):
    """Strict, closed tool arguments: no coercion and no unknown names."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def parameters(model: type[ToolArguments]) -> dict:
    return model.model_json_schema(schema_generator=ToolSchema)
