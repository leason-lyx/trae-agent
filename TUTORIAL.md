# Trae Agent 构建教程

本教程将详细介绍 Trae Agent 的内部工作原理，并指导您如何从零开始理解和构建这样一个基于 LLM 的软件工程智能体。

## 1. 导言：Trae Agent 是什么？

Trae Agent 是一个为通用软件工程任务设计的、基于大语言模型（LLM）的智能体。它通过一个强大的命令行界面（CLI）来理解自然语言指令，并利用多种工具和 LLM 提供商执行复杂的软件工程工作流。

其核心特性包括：
- **多LLM支持**: 可与 OpenAI 和 Anthropic 的 API 配合使用。
- **丰富的工具生态**: 内置文件编辑、代码查看、Bash命令执行、序列化思考等多种工具。
- **交互模式**: 支持通过对话方式进行迭代式开发。
- **轨迹记录**: 详细记录所有 Agent 的行为，便于调试和分析。

## 2. 快速上手

在深入代码之前，我们先确保能够运行它。

### 2.1. 环境准备

- **Python**: 需要 Python 3.12+ 版本。
- **包管理器**: 官方推荐使用 [UV](https://docs.astral.sh/uv/)。
- **API密钥**: 你需要准备 OpenAI 或 Anthropic 的 API 密钥。

### 2.2. 安装与配置

1.  **克隆仓库**:
    ```bash
    git clone <repository-url>
    cd trae-agent
    ```

2.  **安装依赖**:
    ```bash
    uv sync
    ```

3.  **配置API密钥**:
    最简单的方式是设置环境变量：
    ```bash
    # OpenAI
    export OPENAI_API_KEY="your-openai-api-key"

    # Anthropic
    export ANTHROPIC_API_KEY="your-anthropic-api-key"
    ```
    或者，你也可以直接修改 `trae_config.json` 文件。

## 3. Trae Agent 工作原理解析：追踪一次任务之旅

为了真正理解 Trae Agent 的构建方式，我们将扮演侦探的角色，追踪一个典型任务的完整生命周期。假设用户在命令行输入了以下指令：

```bash
trae-cli run "Fix the bug in main.py" --working-dir /app --must-patch
```

这个简单的指令将触发一系列精妙的内部操作。让我们一步步揭开其神秘面纱。

### 3.1. 第一站：命令行入口 (`cli.py`)

所有交互都始于 `cli.py`。这个文件使用 `click` 库创建了一个用户友好的命令行界面。

当用户执行 `run` 命令时，对应的 `run` 函数会被调用：

```python
# trae_agent/cli.py

@cli.command()
@click.argument('task')
@click.option('--working-dir', '-w', help='Working directory for the agent')
@click.option('--must-patch', '-mp', is_flag=True, help='Whether to patch the code')
# ... 其他选项 ...
def run(task: str, working_dir: str | None = None, must_patch: bool = False, ...):
    """Run a task using Trae Agent."""

    # 1. 加载配置
    config = load_config(...)

    # 2. 创建 Agent 实例
    agent: TraeAgent = create_agent(config)

    # 3. 设置轨迹记录
    trajectory_path = agent.setup_trajectory_recording()

    # 4. 准备任务参数
    task_args = {
        "project_path": working_dir,
        "issue": task,
        "must_patch": "true" if must_patch else "false"
    }

    # 5. 初始化并执行任务
    agent.new_task(task, task_args)
    _ = asyncio.run(agent.execute_task())

    console.print(f"\n[green]Trajectory saved to: {trajectory_path}[/green]")
```

这里的逻辑非常清晰：
1.  **加载配置**: `load_config` 函数会整合命令行参数、`trae_config.json` 文件和环境变量，形成一个统一的 `Config` 对象。
2.  **创建 Agent**: `create_agent` 工厂函数实例化了 `TraeAgent`。
3.  **设置轨迹**: `setup_trajectory_recording` 准备好记录 Agent 的所有行为。
4.  **封装参数**: 将任务描述、工作目录等信息打包成一个 `task_args` 字典。
5.  **启动引擎**: 调用 `agent.new_task()` 进行初始化，然后 `asyncio.run(agent.execute_task())` 正式启动 Agent 的核心循环。

### 3.2. 第二站：任务初始化 (`trae_agent.py`)

`agent.new_task()` 是 Agent 执行任务前的准备阶段。它为即将到来的与 LLM 的“对话”构建了初始上下文。

```python
# trae_agent/agent/trae_agent.py

def new_task(self, task: str, extra_args: dict[str, str] | None = None, tool_names: list[str] | None = None):
    self.task: str = task

    # 1. 加载工具
    if tool_names is None:
        tool_names = TraeAgentToolNames # ["str_replace_based_edit_tool", "sequentialthinking", ...]
    self.tools: list[Tool] = [tools_registry[tool_name]() for tool_name in tool_names]
    self.tool_caller: ToolExecutor = ToolExecutor(self.tools)

    # 2. 构建与LLM的初始对话
    self.initial_messages: list[LLMMessage] = []
    
    # 对话第一句：系统提示，定义Agent的角色和规则
    self.initial_messages.append(LLMMessage(role="system", content=self.get_system_prompt()))

    # 对话第二句：用户的具体任务
    user_message = ""
    if extra_args:
        user_message += f"[Project root path]:\n{extra_args['project_path']}\n\n"
        user_message += f"[Problem statement]: ...\n{extra_args['issue']}\n"
    
    self.initial_messages.append(
        LLMMessage(role="user", content=user_message)
    )
```

这里的关键在于 `initial_messages` 的构建，它精确地设定了对话的起点：
- **`role: "system"`**: 这是高优先级的指令。`get_system_prompt()` 的返回内容（我们稍后详述）告诉 LLM 它应该扮演什么角色，以及它必须遵守的工作流程。
- **`role: "user"`**: 这是用户的具体请求，包含了问题描述和工作目录等关键信息。

### 3.3. Agent 的“大脑”：系统提示 (`get_system_prompt`)

这是 Trae Agent 智能行为的蓝图和核心。这个精心设计的提示词，将一个通用的 LLM “改造”成一个专业的软件工程师。它不仅仅是指令，更是一个行为框架。

让我们剖析其核心思想：
> You are an expert AI software engineering agent. Your primary goal is to resolve a given GitHub issue...
> Follow these steps methodically:
> 1.  Understand the Problem...
> 2.  Explore and Locate...
> 3.  **Reproduce the Bug (Crucial Step)**: ... you **must** create a script or a test case that reliably reproduces the bug.
> 4.  Debug and Diagnose...
> 5.  Develop and Implement a Fix...
> 6.  Verify and Test Rigorously...
> 7.  Summarize Your Work...

这个提示词的精妙之处在于：
- **角色扮演**: 赋予 LLM 一个“专家”身份，能激发其在训练数据中学到的更高级的推理能力。
- **流程化**: 将复杂的软件开发任务分解成一个线性的、可执行的步骤序列。这避免了 LLM 天马行空、不着边际的回答。
- **强制约束**: 对“复现 Bug”和“严格测试”的强制要求，注入了测试驱动开发（TDD）和质量保障的工程思想，极大地提升了产出代码的可靠性。

可以说，**系统提示是 Agent 的灵魂，定义了它的性格、能力边界和工作质量的下限。**

### 3.4. 第三站：核心执行循环 (`agent/base.py`)

当 `agent.execute_task()` 被调用时，我们进入了 Agent 的心跳所在——一个不断与 LLM 对话、调用工具、观察结果的循环。这个通用的循环逻辑定义在 `Agent` 基类中。

```python
# trae_agent/agent/base.py

async def execute_task(self) -> AgentExecution:
    # ...
    messages = self.initial_messages
    step_number = 1

    while step_number <= self.max_steps:
        # 1. 调用 LLM (思考)
        llm_response = self.llm_client.chat(messages, self.model_parameters, self.tools)

        # 2. 检查任务是否完成
        if self.llm_indicates_task_completed(llm_response):
            if self.is_task_completed(llm_response):
                # ... 任务成功，跳出循环 ...
                break
            else:
                # ... 未满足 'must_patch' 条件，生成提示信息并继续循环 ...
                messages = [LLMMessage(role="user", content=self.task_incomplete_message())]
        else:
            # 3. 如果未完成，检查是否有工具调用
            tool_calls = llm_response.tool_calls
            if tool_calls and len(tool_calls) > 0:
                # 4. 执行工具 (行动)
                tool_results = await self.tool_caller.sequential_tool_call(tool_calls)
                
                # 5. 将工具结果打包成消息，用于下一次循环
                messages = []
                for tool_result in tool_results:
                    message = LLMMessage(role="user", tool_result=tool_result)
                    messages.append(message)
        
        step_number += 1
    # ...
```

这个 `while` 循环就是著名的“**ReAct**”（Reasoning and Acting）模式的实现：
1.  **思考 (Reason)**: 调用 `self.llm_client.chat()`，将当前对话历史发送给 LLM，获取其下一步的思考和决策 (`llm_response`)。
2.  **行动 (Act)**:
    - 如果 LLM 认为任务完成，则进行最终检查。
    - 如果 LLM 返回了工具调用请求 (`tool_calls`)，则进入下一站：工具执行器。
3.  **观察与反馈**: 工具执行的结果 (`tool_results`) 被格式化成新的消息，成为下一次循环“思考”阶段的输入。这形成了一个完整的闭环，Agent 通过不断地与环境交互来迭代地接近最终目标。

### 3.5. 第四站：工具执行器 (`tools/base.py`)

当 LLM 决定要“做”点什么时，它会生成一个符合预定义格式的工具调用请求。例如：

```xml
<tool_calls>
  <tool_call>
    <name>bash</name>
    <input>
      <command>ls -F /app</command>
    </input>
  </tool_call>
</tool_calls>
```

`agent/base.py` 中的 `self.tool_caller`（一个 `ToolExecutor` 实例）负责接收并处理这个请求。

```python
# trae_agent/tools/base.py

class ToolExecutor:
    def __init__(self, tools: list[Tool]):
        # 将工具列表转换成一个字典，便于通过名称查找
        self.tools: dict[str, Tool] = {tool.name: tool for tool in tools}

    async def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        # 1. 检查工具是否存在
        if tool_call.name not in self.tools:
            return ToolResult(success=False, error=f"Tool '{tool_call.name}' not found.")

        tool = self.tools[tool_call.name]

        try:
            # 2. 调用工具自身的 execute 方法
            tool_exec_result = await tool.execute(tool_call.arguments)
            return ToolResult(
                success=(tool_exec_result.error is None),
                result=tool_exec_result.output,
                error=tool_exec_result.error,
                ...
            )
        except Exception as e:
            # ... 异常处理 ...
```

`ToolExecutor` 像一个调度中心：
1.  它通过工具名称 (`tool_call.name`) 从 `self.tools` 字典中找到对应的工具对象。
2.  然后调用该工具对象的 `execute` 方法，并将 LLM 提供的参数 (`tool_call.arguments`) 传递过去。
3.  最后，它将工具的执行结果（成功或失败、输出或错误信息）包装成一个 `ToolResult` 对象，返回给核心循环，作为对 LLM “行动”的“观察”结果。

这个清晰的分层设计（Agent -> Executor -> Tool）使得添加新工具变得非常简单。

### 3.6. 终点站：深入工具内部

现在，我们的旅程来到了最后一站：工具的实际执行代码。这些代码是 Agent 与真实世界（文件系统、Shell 环境）交互的桥梁。

#### `bash_tool.py`：拥有记忆的终端

一个简单的 `bash` 工具可能会在每次调用时都创建一个新的 Shell 进程。这意味着，如果 LLM 先发送 `cd /app`，再发送 `ls`，第二次的 `ls` 将在新的进程中执行，其当前目录仍然是默认目录，而不是 `/app`。

Trae Agent 通过一个巧妙的设计解决了这个问题：它为每个 Agent 实例维护一个**持久化的 Bash 会话**。

```python
# trae_agent/tools/bash_tool.py

class _BashSession:
    # ...
    async def start(self):
        # 创建一个长活的子进程
        self._process = await asyncio.create_subprocess_shell(...)

    async def run(self, command: str) -> ToolExecResult:
        # ...
        # 将命令写入同一个子进程的 stdin
        self._process.stdin.write(command.encode() + b"\n")
        await self._process.stdin.drain()

        # 从该子进程的 stdout 读取结果
        # ...

class BashTool(Tool):
    def __init__(self):
        # 每个 BashTool 实例拥有一个 _BashSession
        self._session: _BashSession | None = None
        super().__init__()

    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        if self._session is None:
            # 仅在第一次调用时创建并启动会话
            self._session = _BashSession()
            await self._session.start()
        
        # 后续所有命令都在这个已存在的会话中运行
        return await self._session.run(command)
```

通过 `_BashSession` 类，`BashTool` 确保了所有的 `bash` 命令都在同一个底层 Shell 进程中执行。这使得 Agent 的行为更符合人类使用终端的直觉，可以执行 `cd`、设置环境变量等连续性操作。

#### `edit_tool.py`：精密的外科手术刀

文件编辑是软件工程的核心任务。`edit_tool.py` 必须做到既强大又安全。它的核心是 `str_replace`（字符串替换）功能，其设计体现了对精确性的追求。

```python
# trae_agent/tools/edit_tool.py

class TextEditorTool(Tool):
    # ...
    def str_replace(self, path: Path, old_str: str, new_str: str | None) -> ToolExecResult:
        file_content = self.read_file(path)

        # 1. 精确匹配检查
        occurrences = file_content.count(old_str)
        if occurrences == 0:
            raise ToolError(f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}.")
        
        # 2. 唯一性检查
        elif occurrences > 1:
            raise ToolError(f"No replacement was performed. Multiple occurrences of old_str `{old_str}`. Please ensure it is unique")

        # 3. 执行替换
        new_file_content = file_content.replace(old_str, new_str)
        self.write_file(path, new_file_content)

        # ... 返回替换预览 ...
```

这里的两个前置检查至关重要：
1.  **精确匹配**: `old_str` 必须与文件中的某部分完全一致（包括空格和换行符）。这防止了模糊或错误的替换。
2.  **唯一性保证**: 如果 `old_str` 在文件中出现多次，工具会拒绝操作。这强制要求 LLM 在调用工具时，提供足够长的上下文（`old_str`）来唯一地定位要修改的代码块，从而避免了意外修改其他同名变量或函数。

这种“宁可失败，也不做错”的设计哲学，是构建一个可靠的自动化代码编辑工具的关键。

---
至此，我们完整地走过了一个任务从命令行到最终执行的全过程。希望这次深入的旅程能帮助你理解 Trae Agent 的构建细节和其背后的设计思想。

## 4. 运行与观察

### 4.1. 执行任务
你可以通过 `run` 子命令来执行一次性任务：
```bash
# 示例：修复 main.py 中的一个bug
trae-cli run "Fix the bug in main.py" --working-dir /path/to/project
```

### 4.2. 轨迹记录
每次运行后，Agent 会在当前目录下生成一个 `trajectory_*.json` 文件。这个文件是理解 Agent 思考过程的"黑匣子"。它用JSON格式记录了：
- 完整的 LLM 对话历史（prompt, response）。
- 每一步的工具调用及其参数。
- 工具执行返回的结果。

通过分析这个文件，你可以完整地回溯 Agent 的每一步操作，这对于调试 Agent 本身或理解其决策逻辑至关重要。

## 5. 总结与扩展

通过本教程，我们深入了解了 Trae Agent 的设计哲学和实现细节。它不仅仅是一个简单的 LLM 包装器，而是一个集成了"思考框架"（通过系统提示词）和"行动能力"（通过工具集）的复杂系统。

如果你想扩展此 Agent，可以从以下几点入手：
- **添加新工具**: 在 `trae_agent/tools/` 目录下创建一个新的工具类，继承自 `base.py` 中的 `Tool`，并在 `__init__.py` 中注册它。
- **修改系统提示**: 调整 `get_system_prompt` 的内容，可以改变 Agent 的行为模式或赋予它新的能力。
- **支持新的LLM**: 在 `utils/llm_clients.py` 中添加新的 LLM 客户端实现。

希望本教程能帮助你更好地理解和使用 Trae Agent！