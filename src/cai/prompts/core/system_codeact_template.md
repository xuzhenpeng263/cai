<%
    import os
    from cai.util import cli_print_tool_call
    from cai.rag.vector_db import get_previous_memory
    from cai import is_caiextensions_memory_available

    # Get system prompt from agent if provided
    system_prompt = (
        agent.instructions(context_variables)
        if callable(agent.instructions)
        else agent.instructions
    )

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
%>

${system_prompt}

% if ctf_inside is not None:
    % if ctf_inside.lower() == 'true':
您在Docker容器内的目标机器上。除非必要，尽量不要使用网络命令。
    % elif ctf_inside.lower() == 'false':
您在作为Docker容器的目标机器外部。您可以使用nmap等网络命令进行初始侦察。
    % endif
% endif
