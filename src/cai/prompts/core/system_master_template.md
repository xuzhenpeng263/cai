<%
    # 此系统主文档提供了一个模板
    # 用于构建CAI代理流程和系统的
    # 系统提示词。
    #
    # 提示词的结构包含以下
    # 几个部分：
    #
    # 1. 指令：由代理提供，
    #    对应角色详情和行为。
    #
    # 2. 压缩摘要（可选）：AI生成的摘要，
    #    来自之前的对话，用于减少上下文使用
    #
    # 3. 记忆（可选）：记录在向量数据库中的
    #    过去经验，并重新调用以进行
    #    上下文增强。
    #
    # 4. 推理（可选）：利用推理型
    #    LLM模型（可能与选定的不同）
    #    通过额外的思考过程进一步
    #    增强上下文
    #
    # 5. 环境：执行环境的详细信息，
    #    包括操作系统、IP等。
    #

    import os
    from cai.util import cli_print_tool_call
    try:
        from cai.rag.vector_db import get_previous_memory
    except Exception as e:
        # Silently ignore if RAG module is not available
        pass
    from cai import is_caiextensions_memory_available
    
    # Import compact summary function
    try:
        from cai.repl.commands.memory import get_compacted_summary
        # Get agent name from the agent object
        agent_name = getattr(agent, 'name', None)
        compacted_summary = get_compacted_summary(agent_name)
    except Exception as e:
        compacted_summary = None

    # Get system prompt from the base instructions passed to the template
    # The base instructions are passed as 'ctf_instructions' in the render context
    # We use the pre-set system_prompt variable which equals base_instructions
    # Do NOT call agent.instructions here as that would create infinite recursion!

    # Get CTF_INSIDE environment variable
    ctf_inside = os.getenv('CTF_INSIDE')
    env_context = os.getenv('CAI_ENV_CONTEXT', 'true').lower()
    # Get memory from vector db if RAG is enabled
    rag_enabled = os.getenv("CAI_MEMORY", "?").lower() in ["episodic", "semantic", "all"]
    memory = ""
    if rag_enabled:
        if os.getenv("CAI_MEMORY", "?").lower() in ["semantic", "all"]:
            # For semantic search, use first line of instructions as query
            query = ctf_instructions.split('\n')[0].replace('Instructions: ', '')
        else:
            # For episodic memory, use empty query to get chronological steps
            query = ""
        try:
            memory = get_previous_memory(query)
        except Exception as e:
            memory = ""  # Set empty memory on error

        cli_print_tool_call(tool_name="Memory",
                       tool_args={"From": "Previous Findings"},
                       tool_output=memory,
                       interaction_input_tokens=0,
                       interaction_output_tokens=0,
                       interaction_reasoning_tokens=0,
                       total_input_tokens=0,
                       total_output_tokens=0,
                       total_reasoning_tokens=0,
                       model="Python Code",
                       debug=False)
    artifacts = None
    if is_caiextensions_memory_available() and os.getenv('CTF_NAME'):
        from caiextensions.memory import get_artifacts
        artifacts = get_artifacts(os.getenv('CTF_NAME').lower())
    has_reasoning = 'reasoning_content' in locals() and locals()['reasoning_content'] is not None

%>
${system_prompt}
% if compacted_summary:

<compacted_context>
这是之前对话上下文的摘要，已压缩以节省令牌：

${compacted_summary}

使用此摘要来理解上下文并从对话中断的地方继续。
</compacted_context>
% endif
% if rag_enabled:

<memory>
${memory}
</memory>

请记住，您必须基于提供的记忆，遵循自主执行工具和命令的迭代过程。虽然CTF从头开始，但这是完全相同的场景 - 记忆显示了之前成功完成的步骤。网络状态和侦察信息保持相同，因此避免不必要地重复侦察步骤。相反，专注于以有条理的方式重现关键工具调用和利用步骤以达到相同的目标。例如，如果之前找到了标志，请执行再次获取该标志所需的特定命令。始终使用标志判别器来验证标志。在遵循记忆中显示的已验证路径的同时，保持工具的持续自主执行。
% endif

% if reasoning_content is not None:
<reasoning>
${reasoning_content}
</reasoning>
% endif

% if env_context.lower() == 'true':
<%
    import platform
    import socket
    import os

    from pathlib import Path

    # Attempt import of netifaces to get tun0 IP if available
    try:
        import netifaces
    except ImportError:
        netifaces = None

    # Gather system info
    try:
        hostname = socket.gethostname()
        ip_addr = socket.gethostbyname(hostname)
        os_name = platform.system()
    except:
        hostname = "local0"
        ip_addr = "127.0.0.1"
        os_name = "Linux"   

    # Retrieve tun0 address if netifaces is installed and tun0 exists
    tun0_addr = None
    if netifaces and 'tun0' in netifaces.interfaces():
        addrs = netifaces.ifaddresses('tun0')
        if netifaces.AF_INET in addrs:
            tun0_addr = addrs[netifaces.AF_INET][0].get('addr', None)

    # Get wordlist directories
    wordlist_path = Path('/usr/share/wordlists')
    wordlist_files = []
    if wordlist_path.exists():
        wordlist_files = [
            f.name for f in wordlist_path.iterdir() if f.is_file()
        ]

    seclists_path = wordlist_path / 'seclists'
    seclist_dirs = []
    if seclists_path.exists():
        seclist_dirs = [
            d.name for d in seclists_path.iterdir() if d.is_dir()
        ]
%>
环境上下文（以"树"格式显示）：
seclists
% if seclist_dirs:
% for dir in seclist_dirs:
├── ${dir}
% endfor
% else:
└── （在seclists中未找到目录）
% endif

- 如有疑问，请再次列出。

攻击者机器信息：
├── 操作系统: ${os_name}
├── 主机名: ${hostname}
├── 攻击者IP（默认）: ${ip_addr}
% if tun0_addr:
├── tun0 IP: ${tun0_addr}
% endif
└── 角色: 攻击者

% if wordlist_files:
可用字典列表 (/usr/share/wordlists):
% for file in wordlist_files:
├── ${file}
% endfor
% endif
% endif

% if artifacts:
有用信息: ${artifacts}
% endif
