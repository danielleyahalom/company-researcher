from langchain_core.messages import AIMessage
from ..classes import ResearchState, TavilySearchInput, TavilyQuery, ReportEvaluation
from langchain_anthropic import ChatAnthropic


class EvaluationNode:
    def __init__(self):
        self.model = ChatAnthropic(
            model="claude-sonnet-5"
        )

    # Evaluation function assigns an overall grade from 1 to 3.
    async def evaluate_report(self, state: ResearchState):
        """
        Evaluates the generated report by assigning an overall grade from 1 to 3.
        If the grade is 1, includes critical gaps in the output.
        """
        prompt = f"""
            You have created a report on '{state['company']}' based on the gathered information.
            Grade the report on a scale of 1 to 3 based on completeness, accuracy, and depth of information:
            - **3** indicates a thorough and well-supported report with no major gaps.
            - **2** indicates adequate coverage, but could be improved.
            - **1** indicates significant gaps or missing essential sections.

            If the grade is 1, specify any critical gaps that need addressing.
            
            Here is the report for evaluation:
            {state['report']}
        """

        # Invoke the model for report evaluation

        messages = ["system","Your task is to evaluate a report on a scale of 1 to 3.",
                ("human",f"{prompt}")]
        evaluation = await self.model.with_structured_output(ReportEvaluation).ainvoke(messages)
        
        # Determine if additional questions are needed based on grade
        if evaluation.grade == 1:
            msg = f"❌ The report received a grade of 1. Critical gaps identified: {', '.join(evaluation.critical_gaps or ['None specified'])}"
            # Create new sub-questions for critical gaps
            new_sub_queries = [
                TavilyQuery(query=f"Gather information on {gap} for {state['company']}", topic="general", days=30)
                for gap in evaluation.critical_gaps or []
            ]
            if 'sub_questions' in state:
                state['sub_questions'].sub_queries.extend(new_sub_queries)
            else:
                state['sub_questions'] = TavilySearchInput(sub_queries=new_sub_queries)
            return {"messages": [AIMessage(content=msg)], "eval": evaluation, "sub_questions": state['sub_questions']}
        else:
            msg = f"✅ The report received a grade of {evaluation.grade}/3 and is marked as complete."
            return {"messages": [AIMessage(content=msg)], "eval": evaluation}

    async def run(self, state: ResearchState):
        result = await self.evaluate_report(state)
        return result
