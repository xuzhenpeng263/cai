"""
LLM as a Judge Pattern

Uses one LLM to perform a task and another to evaluate the result and provide feedback.
The loop continues until the evaluator is satisfied. This improves output quality
and allows cost optimization by combining smaller and larger models.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Literal
from cai.sdk.agents import Agent, ItemHelpers, Runner, TResponseInputItem, OpenAIChatCompletionsModel
from openai import AsyncOpenAI


# CTF task planner agent (performs planning)
ctf_task_planner = Agent(
    name="CTF Task Planner",
    description="Agent focused on creating a task plan to approach a CTF challenge.",
    instructions=(
        "You are a cybersecurity strategist. Given a CTF challenge description, "
        "generate a clear and effective plan of tasks to solve it. "
        "Use any feedback to improve your planning."
    ),
    model=OpenAIChatCompletionsModel(
        model= os.getenv('CAI_MODEL', "qwen2.5:14b"),
        openai_client=AsyncOpenAI(),
    ),
    tools =[]
)


# Feedback structure for the judge
@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]


# CTF task plan evaluator (judges planning quality)
ctf_plan_evaluator = Agent[None](
    name="CTF Plan Evaluator",
    description="Agent that evaluates CTF task plans for effectiveness and completeness.",
    instructions=(
        "You evaluate a task plan created for solving a CTF challenge. "
        "Ensure it covers all essential steps (recon, exploitation, post-exploitation, etc.). "
        "Provide actionable feedback. Never approve on the first try."
    ),
    model=OpenAIChatCompletionsModel(
        model= os.getenv('CAI_MODEL', "qwen2.5:14b"),
        openai_client=AsyncOpenAI(),
    ),
    output_type=EvaluationFeedback,
)


async def main() -> None:
    challenge_desc = input("Describe the CTF challenge: ")
    input_items: list[TResponseInputItem] = [{"content": challenge_desc, "role": "user"}]

    latest_plan: str | None = None
    
    while True:
            planner_result = await Runner.run(ctf_task_planner, input_items)
            input_items = planner_result.to_input_list()
            latest_plan = ItemHelpers.text_message_outputs(planner_result.new_items)
            print("CTF task plan generated.")

            eval_result = await Runner.run(ctf_plan_evaluator, input_items)
            feedback: EvaluationFeedback = eval_result.final_output
            print(f"Evaluator score: {feedback.score}")

            if feedback.score == "pass":
                print("CTF task plan approved.")
                break

            print("Refining plan based on evaluator feedback...")
            input_items.append({"content": f"Feedback: {feedback.feedback}", "role": "user"})

    print(f"Final CTF task plan:\n{latest_plan}")


if __name__ == "__main__":
    asyncio.run(main())