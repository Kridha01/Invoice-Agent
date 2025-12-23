import pandas as pd
import json
from typing import Dict, Optional, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from rag import FieldExtractor



class JudgeFeedback(BaseModel):
    feedback: Dict[str, str] = Field(
        description="Field-wise feedback to improve the next extraction attempt"
    )

def get_judge_chain():
    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    judge_parser = PydanticOutputParser(pydantic_object=JudgeFeedback)
    judge_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert invoice auditor.
            Compare extracted values with ground truth.
            Return ONLY structured feedback for each field.
            {format_instructions}
            """
                ),
                (
                    "human",
                    """
            Fields: {fields}
            Extracted values: {extracted}
            Ground truth: {ground_truth}
            """
                )
            ])
    judge_chain=(judge_prompt | judge_llm | judge_parser)
    return judge_chain,judge_parser


def evaluate_df(df: pd.DataFrame, gt_map: Dict[str, str]):
    correct = 0
    feedback = {}

    for field, gt_value in gt_map.items():
        pred = str(df[field].iloc[0]).strip()
        gt = str(gt_value).strip()

        if pred == gt:
            correct += 1
        else:
            feedback[field] = f"Expected '{gt}', but got '{pred}'"

    score = (correct / len(gt_map)) * 100
    return score, feedback


class AgentState(BaseModel):
    extractor: FieldExtractor
    pdf_path: str
    excel_path: str
    gt_map: Dict[str, str]
    iteration: int = 0
    max_iterations: int = 1
    threshold: float = 97.0
    current_df: Optional[pd.DataFrame] = None
    best_df: Optional[pd.DataFrame] = None
    best_score: float = 0.0
    score: float = 0.0
    feedback: Dict[str, str] = Field(default_factory=dict)
    model_config = {
        "arbitrary_types_allowed": True  
    }

def extract_node(state: AgentState) -> AgentState:
    df = state.extractor.extract_fields() 
    # print("*"*20)
    # print(df)
    state.current_df = df
    state.best_df=df
    return state

def evaluate_node(state: AgentState) -> AgentState:
    score, feedback = evaluate_df(state.current_df, state.gt_map)

    state.score = score
    # print(f"score for iteratin {state.iteration},{state.score}")
    state.feedback = feedback
    if score > state.best_score:
        state.best_score = score
        state.best_df = state.current_df

    return state

def optimize_node(state: AgentState) -> AgentState:
   
    # chain,judge_parser = get_judge_chain()
    # print("chain", chain)
    # print("parser",judge_parser)
    fields = list(state.current_df.columns)
    extracted = state.current_df.iloc[0].to_dict()
    ground_truth = state.gt_map  

    judge_chain, judge_parser = get_judge_chain()

    messages = judge_chain.first.format_messages(
        fields=fields,
        extracted=extracted,
        ground_truth=ground_truth,
        format_instructions=judge_parser.get_format_instructions()
    )
    # print(messages)
    # print("\n====== FINAL PROMPT SENT TO LLM ======\n")
    
    # for msg in messages:
    #     print(f"[{msg.type.upper()}]")
    #     print(msg.content)
    #     print("------------------------------------")

    judge_result: JudgeFeedback = judge_chain.invoke({
        "fields": fields,
        "extracted": extracted,
        "ground_truth": ground_truth,
        "format_instructions": judge_parser.get_format_instructions()
    })

    # print("Judge Result")
    # print(judge_result)
    state.feedback = judge_result.feedback
    state.iteration += 1
    # print("state iteration",state.iteration)
    # print("feedback",state.feedback)
    return state


def route_after_evaluation(state: AgentState):
    if state.score >= state.threshold:
        return "final"
    if state.iteration >= state.max_iterations:
        return "final"
    return "optimize"

def final_node(state: AgentState):
    state.current_df = state.best_df
    # print(state.best_df)
    return state

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("extract", extract_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("optimize", optimize_node)
    graph.add_node("final", final_node)

    graph.set_entry_point("extract")

    graph.add_edge("extract", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {
            "optimize": "optimize",
            "final": "final"
        }
    )

    graph.add_edge("optimize", "extract")
    graph.add_edge("final", END)

    return graph.compile()

def run_agent(pdf_path, excel_path):
    extractor = FieldExtractor(pdf_path, excel_path, api_key="")
    gt_map = extractor.get_ground_truth_map()
    graph = build_graph()

    state = AgentState(
        pdf_path=pdf_path,
        excel_path=excel_path,
        extractor=extractor,
        gt_map=gt_map
    )

    final_state = graph.invoke(state)
    print(final_state)

    # print("final_df")
    # print(final_state['best_df'])
    return final_state['best_df']


# if __name__ == "__main__":
#     pdf_path = "sample-invoice.pdf"   
#     excel_path = "fields.xlsx"        
#     final_df = run_agent(pdf_path, excel_path)
    
#     final_df.to_excel("final_output.xlsx", index=False)
#     print("Saved final output to final_output.xlsx")