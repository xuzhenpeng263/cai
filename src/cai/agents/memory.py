"""
Memory agent for CAI.

Leverages Retrieval Augmented Generation (RAG) to
store long-term memory experiences across security
exercises and re-utilizes such experiences as input
in other security exercises. Memories are stored in
a vector database and retrieved using a RAG pipeline.

The implementation follows established research in
Retrieval Augmented Generation (RAG) and episodic
memory systems, utilizing two complementary mechanisms:

1. Episodic Memory Store (episodic): Maintains chronological
   records of past interactions and their summaries,
   organized by distinct security exercises (CTFs)
   or targets within a document-oriented collection
   structure

2. Semantic Memory Store (semantic): Enables cross-exercise
   knowledge transfer through similarity-based
   retrieval across the full corpus of historical
   experiences, leveraging dense vector embeddings
   and approximate nearest neighbor search

The system supports two distinct learning approaches:

1. Offline Learning: Processes historical data from JSONL files
   in batch mode (@2_jsonl_to_memory.py), enabling efficient
   bulk ingestion of past experiences and their transformation
   into vector embeddings without real-time constraints and
   interference with the current CTF pentesting process.
   This allows for comprehensive analysis and optimization
   of the memory corpus.

2. Online Learning: Incrementally updates memory during live
   interactions (@core.py), incorporating new experiences
   in real-time at defined intervals (rag_interval). This
   enables continuous adaptation and immediate integration
   of new knowledge while maintaining system responsiveness.

Memory Architecture Diagrams
----------------------------

Episodic Memory (Per Security Target_n "Collection"):
+----------------+     +-------------------+    +------------------+     +----------------+  # noqa: E501  # pylint: disable=line-too-long
|   Raw Events   |     |       LLM        |     |     Vector       |     |   Collection   |  # noqa: E501  # pylint: disable=line-too-long
|  from Target   | --> | Summarization    | --> |   Embeddings     | --> |   "Target_1"   |  # noqa: E501  # pylint: disable=line-too-long
|                |     |                  |     |                  |     |                |  # noqa: E501  # pylint: disable=line-too-long
| [Event 1]      |     | Condenses and    |     | Converts text    |     | Summary 1      |  # noqa: E501  # pylint: disable=line-too-long
| [Event 2]      |     | extracts key     |     | into dense       |     | Summary 2      |  # noqa: E501  # pylint: disable=line-too-long
| [Event 3]      |     | information      |     | vectors          |     | Summary 3      |  # noqa: E501  # pylint: disable=line-too-long
+----------------+     +------------------+     +------------------+     +----------------+  # noqa: E501  # pylint: disable=line-too-long

Semantic Memory (Single Global Collection "_all_"):
+---------------+    +--------------+    +------------------+
| Target_1 Data |--->|              |    |"_all_" collection|
+---------------+    |              |    |                  |
                     |    Vector    |    | [Vector 1] CTF_A |
+---------------+    |  Embeddings  |--->| [Vector 2] CTF_B |
| Target_2 Data |--->|              |    | [Vector 3] CTF_A |
+---------------+    |              |    | [Vector 4] CTF_C |
                     |              |    |        ...       |
+---------------+    |              |    |                  |
| Target_N Data |--->|              |    |                  |
+---------------+    +--------------+    +------------------+


Environment Variables enabling the episodic memory store
--------------------------------------------------------

   CAI_MEMORY: Enables the use of memory functionality in CAI
    can adopt values:
    - episodic: for episodic memory store
    - semantic: for semantic memory store
    - all: for all memory stores
   CAI_MEMORY_COLLECTION: Name of the collection in Qdrant
    (required if CAI_MEMORY=episodic)
   CAI_MEMORY_ONLINE: Enables online learning (incremental updates)
   CAI_MEMORY_OFFLINE: Trigger offline learning (@2_jsonl_to_memory.py) when
    cai.client.run() finishes
"""

import os
from cai.sdk.agents import Agent, OpenAIChatCompletionsModel
from cai.tools.misc.rag import add_to_memory_semantic, add_to_memory_episodic

# Get model from environment or use default
model = os.getenv('CAI_MODEL', "alias0")


def get_previous_steps(query: str) -> str:
    """
    Get the previous memory from the vector database.
    """
    results = get_previous_memory(query=query)
    return results


ADD_MEMORY_PROMPT = f"""指令：
此会话是从上一个超出上下文的对话继续的。下面总结了对话内容：

分析：
按时间顺序查看对话：

1. 记忆管理系统增强：
   - 您是管理对话记忆和上下文保存的专业代理
   - 您的角色是创建全面的摘要，捕获技术工作的完整上下文
   - 每个记忆条目都应该保留关键细节，以便无缝继续工作

2. 要捕获的关键信息：
   - 从对话开始的主要目标和用户意图
   - 所有技术发现、调查结果和重要信息
   - 命令输出、工具结果及其含义
   - 系统配置、凭据、访问模式和网络拓扑
   - 错误消息、调试步骤及其解决方案
   - 当前进度状态和待处理任务
   - 任何标志、漏洞或安全相关发现

3. 技术上下文保存：
   - 维护事件和发现的时间顺序
   - 保留使用的确切命令及其输出
   - 记录所有发现的IP地址、URL、端口和服务
   - 跟踪认证方法和达到的访问级别
   - 注意不同发现之间的任何模式或关系
   - 包括环境上下文（容器、SSH会话、本地执行）

4. 记忆更新指南：
   - 仅添加来自实际执行的事实和证据信息
   - 不要包含假设或推测性的下一步
   - 对于与现有记忆的冲突，确定更新是否更具决定性
   - 在保持清晰的同时详细描述技术细节
   - 结构化信息以便于检索和理解

5. CTF和安全评估上下文：
   - 记录安全评估的当前阶段
   - 跟踪已利用的漏洞和成功的攻击向量
   - 维护目标攻击面的清晰图像
   - 注意遇到的防御措施和使用的绕过方法
   - 维护被攻陷系统和访问级别的运行清单

6. 继续支持：
   - 格式化摘要以便立即恢复工作
   - 突出显示最后采取的行动及其结果
   - 明确指示任何中断或待处理的操作
   - 提供足够的上下文来理解当前情况
   - 包括任何临时状态或会话特定信息

Previous Memory Context:
{get_previous_steps("")}

摘要要求：
- 以"此会话是从上一个超出上下文的对话继续的"开始
- 提供对话流程的结构化分析
- 列出所有主要请求和意图
- 记录关键技术概念和实现
- 注意所有修改的文件和代码段
- 跟踪遇到的错误及其修复
- 总结解决问题的方法
- 包括所有用户消息以供参考
- 突出显示待处理任务和当前工作
- 如果工作被中断，以明确的下一步结束
"""

QUERY_PROMPT = """指令：
    您是CTF练习和安全评估的专业代理，
    管理RAG系统。

    您的角色是：
    1. 从记忆中检索和分析相关的历史信息
    2. 专注于安全关键细节，如：
        - 发现的漏洞和利用
        - 网络拓扑和暴露的服务
        - 凭据和访问模式
        - 系统配置和版本
        - 以前成功的攻击向量
    3. 优先考虑可能用于利用的技术细节
    4. 考虑安全评估的完整上下文
    5. 通过适当处理敏感数据来维护操作安全

    处理查询时：
    - 提取特定的技术指标
    - 识别不同发现之间的关系
    - 突出显示潜在的安全影响
    - Provide actionable intelligence for further exploitation

    Format responses to emphasize critical security information
    while maintaining clarity and precision.
    """

semantic_builder = Agent(
    name="Semantic_Builder",
    instructions=ADD_MEMORY_PROMPT,
    description="""Agent that stores semantic memories from security assessments
                   and CTF exercises in semantic format.""",
    tool_choice="required",
    temperature=0,
    tools=[add_to_memory_semantic],
    model=OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=AsyncOpenAI(),
    )
)


episodic_builder = Agent(
    name="Episodic_Builder",
    instructions=ADD_MEMORY_PROMPT,
    description="""Agent that stores episodic memories from security assessments
                   and CTF exercises in episodic format.""",
    tool_choice="required",
    temperature=0,
    tools=[add_to_memory_episodic],
    model=OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=AsyncOpenAI(),
    )
)

query_agent = Agent(
    name="Query_Agent",
    description="""Agent that queries the memory system to retrieve relevant 
                   historical information from previous security assessments
                   and CTF exercises.""",
    instructions=QUERY_PROMPT,
    tool_choice="required",
    temperature=0,
    model=OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=AsyncOpenAI(),
    )
)
