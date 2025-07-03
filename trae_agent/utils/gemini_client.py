"""Gemini API client wrapper with tool integration."""

import os
import json
import random
import time
from google import genai
from google.genai import types
from typing import override

from ..tools.base import Tool, ToolCall, ToolResult
from ..utils.config import ModelParameters
from .base_client import BaseLLMClient
from .llm_basics import LLMMessage, LLMResponse, LLMUsage


class GeminiClient(BaseLLMClient):
    """OpenAI client wrapper with tool schema generation."""

    def __init__(self, model_parameters: ModelParameters):
        super().__init__(model_parameters)

        if self.api_key == "":
            self.api_key: str = os.getenv("GEMINI_API_KEY", "")

        if self.api_key == "":
            raise ValueError("GEMINI API key not provided. Set GEMINI_API_KEY in environment variables or config file.")

        self.client : genai.Client = genai.Client(api_key=self.api_key)
        self.message_history : list[types.Content] = []
        self.system_message: str | None = None

    @override
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history = self.parse_messages(messages)

    @override
    def chat(self, messages: list[LLMMessage], model_parameters: ModelParameters, tools: list[Tool] | None = None, reuse_history: bool = True) -> LLMResponse:
        """Beginning chat messages to gemini with optional tool support."""
        # print("-"*10)
        # print("Sending messages to Gemini API...")
        gemini_messages: list[types.Content] = self.parse_messages(messages)
        # print(f"Parsed input messages: {gemini_messages}")
        if reuse_history:
            self.message_history = self.message_history + gemini_messages
        else:
            self.message_history = gemini_messages
        
        # Add tools to the request if available
        function_declarations: list[types.FunctionDeclaration] = []
        if tools:
            for tool in tools:
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.get_input_schema()
                    )
                )
        config: types.GenerateContentConfig = types.GenerateContentConfig(
            system_instruction = self.system_message,
            temperature=model_parameters.temperature,
            top_p=model_parameters.top_p,
            top_k=model_parameters.top_k,
            max_output_tokens=model_parameters.max_tokens,
            tools = [types.Tool(function_declarations=function_declarations)]
        )
        response = None
        error_message = ""
        for i in range(model_parameters.max_retries):
            try:
                # print("full input contents:", self.message_history)
                response = self.client.models.generate_content(
                    model=model_parameters.model,
                    contents=self.message_history,
                    config=config,
                )
                break
            except Exception as e:
                error_message += f"Error {i + 1}: {str(e)}\n"
                # print(f"Error in Gemini API call: {error_message}")
                # Randomly sleep for 3-30 seconds
                time.sleep(random.randint(3, 30))
                continue

        if response is None:
            raise ValueError(f"Failed to get response from Anthropic after max retries: {error_message}")
        
        # Parse the response
        content = ""
        tool_calls: list[ToolCall] = []
        
        # Collect tool calls from the response
        if response.function_calls is not None and len(response.function_calls) > 0:
            for fn in response.function_calls:
                if fn.name:
                    tool_calls.append(
                        ToolCall(
                            id=fn.id,
                            call_id=fn.id,
                            name=fn.name,
                            arguments=fn.args
                        )
                    )
                    self.message_history.append(
                        types.Content(
                            role="model",
                            parts=[types.Part(function_call=fn)]
                        )
                    )
                else:
                    raise ValueError("Function call name is required")
        # print(f"Gemini output tool calls: {tool_calls}")

        # Collect text content from the response
        if response.candidates and len(response.candidates) > 0 and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.text:
                    content += part.text
                    self.message_history.append(
                        types.Content(
                            role="model",
                            parts=[types.Part(text=part.text)]
                        )
                    )
        
        # print(f"Gemini output response text content: {content}")
        
        usage = None
        if response.usage_metadata:
            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0
            cache_read_tokens = response.usage_metadata.cached_content_token_count or 0
            reasoning_tokens = response.usage_metadata.thoughts_token_count or 0
            usage = LLMUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens = 0,
                cache_read_input_tokens=cache_read_tokens,
                reasoning_tokens=reasoning_tokens,
            )

        llm_response = LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            model=response.model_version,
            finish_reason=response.candidates[0].finish_reason,)
        
        # Record trajectory if recorder is available
        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_interaction(
                messages=messages,
                response=llm_response,
                provider="gemini",
                model=model_parameters.model,
                tools=tools
            )
        
        # print(f"final response: {llm_response}")
        # print()
        return llm_response

    @override
    def supports_tool_calling(self, model_parameters: ModelParameters) -> bool:
        """Check if the current model supports tool calling."""

        if "gemini-2.5" in model_parameters.model:
            return True

        tool_capable_models = [
            "gemini-2.0-flash",
        ]
        return any(model in model_parameters.model for model in tool_capable_models)

    def parse_messages(self, messages: list[LLMMessage]) -> list[types.Content]:
        """Parse the messages to gemini format."""
        # print(f"Parsing messages: {messages}")
        gemini_messages: list[types.Content] = []
        for msg in messages:
            if msg.tool_result:
                gemini_messages.append(
                    types.Content(
                        role="user",
                        parts=[self.parse_tool_call_result(msg.tool_result)],
                    )
                )
            elif msg.tool_call:
                gemini_messages.append(
                    types.Content(
                        role="model",
                        parts=[self.parse_tool_call(msg.tool_call)],
                    )
                )
            else:
                if not msg.content:
                    raise ValueError("Message content is required")
                if msg.role == "system":
                    self.system_message = msg.content if msg.content else None
                elif msg.role == "user":
                    gemini_messages.append(
                        types.Content(
                            role="user",
                            parts=[types.Part(text=msg.content)],
                        )
                    )
                elif msg.role == "assistant":
                    gemini_messages.append(
                        types.Content(
                            role="model",
                            parts=[types.Part(text=msg.content)],
                        )
                    )
                else:
                    raise ValueError(f"Invalid message role: {msg.role}")
        # print("gemini_messages: ",gemini_messages)
        return gemini_messages

    def parse_tool_call(self, tool_call: ToolCall) -> types.Part:
        """Parse the tool call from the LLM response."""
        return types.Part.from_function_call(
            name=tool_call.name,
            args=tool_call.arguments,
        )

    def parse_tool_call_result(self, tool_call_result: ToolResult) -> types.Part:
        """Parse the tool call result from the LLM response."""
        response : dict = {}
        if not tool_call_result.name:
            tool_call_result.name = "name"  # Default name if not provided
            # raise ValueError("ToolResult.name 不能为空")
        if tool_call_result.result:
            response["result"] = tool_call_result.result # SDK源代码建议这里是output字段，但是文档是result字段
        if tool_call_result.error:
            response["error"] = tool_call_result.error
        return types.Part.from_function_response(name = tool_call_result.name, response=response)