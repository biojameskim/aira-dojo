# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Callable
from omegaconf import DictConfig

from dojo.core.solvers.llm_helpers.generic_llm import GenericLLM
from dojo.core.solvers.utils.response import wrap_code
from dojo.core.solvers.utils.journal import Node, Journal


judge_schema = """{
    "type": "object",
    "properties": {
        "priors": {
            "type": "array",
            "items": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "description": "Probability distribution over children indicating which solutions are most promising for deeper exploration. Must sum to 1.0. Higher values indicate more exploration priority."
        },
        "reasoning": {
            "type": "string",
            "description": "Overall reasoning explaining the probability distribution and which approaches are most promising to explore deeper."
        },
        "child_assessments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "Index of the child being assessed"
                    },
                    "assessment": {
                        "type": "string",
                        "description": "Detailed assessment of this child's promise"
                    },
                    "promising": {
                        "type": "boolean",
                        "description": "Whether this solution is promising for deeper exploration"
                    },
                    "concerns": {
                        "type": "string",
                        "description": "Any concerns or limitations with this approach"
                    }
                },
                "required": ["id", "assessment", "promising", "concerns"]
            },
            "description": "Detailed assessment for each child solution"
        }
    },
    "required": ["priors", "reasoning", "child_assessments"]
}"""


def judge_op(
    judge_llm: GenericLLM,
    cfg: DictConfig,
    memory_op: Optional[Callable[[Journal, Optional[Node]], str]],
    task_description: str,
    parent_node: Node,
    children: List[Node],
    journal: Journal,
    current_step: int,
    data_preview: Optional[str] = None,
) -> str:
    """
    Judge which children solutions are most promising for deeper exploration.

    This operator compares all children of a parent node and assigns a probability
    distribution (priors) indicating which directions are worth exploring more deeply.

    Args:
        judge_llm: GenericLLM instance for judging
        cfg: Solver configuration
        memory_op: Memory operator for retrieving search history context
        task_description: The task being solved
        parent_node: Parent node whose children are being judged
        children: List of child nodes to judge (all executed with metrics)
        journal: Full search history
        current_step: Current iteration in the search
        data_preview: Optional preview of the data

    Returns:
        Tuple of (json_output, metrics_dict) containing priors and reasoning
    """
    # Prepare parent context
    parent_context = {
        "code": wrap_code(parent_node.code) if parent_node.code else "No code (root node)",
        "plan": parent_node.plan if parent_node.plan else "Initial state",
        "metric": parent_node.metric.value,  # None for WorstMetricValue
        "execution_output": wrap_code(parent_node.term_out, lang=""),
        "analysis": parent_node.analysis if parent_node.analysis else "No analysis",
        "is_buggy": parent_node.is_buggy,
    }

    # Prepare children context
    children_context = []
    for idx, child in enumerate(children):
        child_ctx = {
            "id": idx,
            "code": wrap_code(child.code),
            "plan": child.plan,
            "operator": child.operators_used[0] if child.operators_used else "unknown",
            "metric": child.metric.value,  # None for WorstMetricValue
            "execution_output": wrap_code(child.term_out, lang=""),
            "analysis": child.analysis if child.analysis else "No analysis",
            "is_buggy": child.is_buggy,
            "exec_time": child.exec_time,
        }
        children_context.append(child_ctx)

    # Get memory/history context
    memory_context = ""
    if memory_op:
        memory_context = memory_op(journal, parent_node)

    # Get best metric so far
    best_node = journal.get_best_node()
    best_metric = best_node.metric.value if best_node else None

    # Build judge data
    judge_data = {
        "task_desc": task_description,
        "data_preview": data_preview if data_preview else "No data preview available",
        "parent": parent_context,
        "children": children_context,
        "num_children": len(children),
        "memory": memory_context,
        "current_step": current_step,
        "best_metric_so_far": best_metric,
    }

    return judge_llm(
        query_data=judge_data,
        json_schema=judge_schema,
        function_name="judge_solutions",
        function_description="Judge which child solutions are most promising for deeper exploration in the search tree. Assign a probability distribution over children indicating exploration priority.",
        no_user_message=True,
    )
