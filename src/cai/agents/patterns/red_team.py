"""
Implementation of a Cyclic Swarm Pattern for Red Team Operations

This module establishes a coordinated multi-agent system where specialized agents
collaborate on security assessment tasks. The pattern implements a directed graph
of agent relationships, where each agent can transfer context (message history) 
to another agent through handoff functions, creating a complete communication network
for comprehensive security analysis.
"""
from cai.agents.red_teamer import redteam_agent
from cai.agents.thought import thought_agent
from cai.agents.mail import dns_smtp_agent
from cai.sdk.agents import handoff


# Create handoffs using the SDK handoff function
dns_smtp_handoff = handoff(
    agent=dns_smtp_agent,
    tool_description_override="Use for DNS scans and domain reconnaissance about DMARC and DKIM records"
)

redteam_handoff = handoff(
    agent=redteam_agent,
    tool_description_override="Transfer to Red Team Agent for security assessment and exploitation tasks"
)

thought_handoff = handoff(
    agent=thought_agent,
    tool_description_override="Transfer to Thought Agent for analysis and planning"
)

# Register handoff to enable inter-agent communication pathways
redteam_agent.handoffs.append(dns_smtp_handoff)
dns_smtp_agent.handoffs.append(redteam_handoff)
thought_agent.handoffs.append(redteam_handoff)

# Initialize the swarm pattern with the thought agent as the entry point
redteam_swarm_pattern = thought_agent
redteam_swarm_pattern.pattern = "swarm"