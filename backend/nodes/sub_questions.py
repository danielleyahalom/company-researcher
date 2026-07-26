from langchain_core.messages import AIMessage
from langchain_anthropic import ChatAnthropic
from ..classes import ResearchState, TavilySearchInput

class SubQuestionsNode:
    def __init__(self) -> None:
        self.model = ChatAnthropic(
            model="claude-haiku-4-5-20251001"
        )
     
    # Function to generate sub-questions based on initial search data
    async def generate_sub_questions(self, state: ResearchState):
        try:
            msg = "🤔 Generating sub-questions based on the initial search results...\n"
            
            if 'sub_questions_data' not in state:
                state['sub_questions_data'] = []
                
            # Prompt to generate detailed sub-questions
            prompt = f"""
            You are an expert researcher focusing on company analysis to generate a report.
            Your task is to generate 4 specific sub-questions that will provide a thorough understanding of the company: '{state['company']}'.
            
            ### Key Areas to Explore:
            - **Company Background**: Include history, mission, headquarters location, CEO, and number of employees.
            - **Products and Services**: Focus on main offerings, unique features, and target customer segments.
            - **Market Position**: Address competitive standing, market reach, and industry impact.
            - **Financials**: Seek recent funding, revenue milestones, financial performance, and growth indicators.

            Use the initial information provided from the company's website below to keep questions directly relevant to **{state['company']}**.

            Official URL: {state['company_url']}
            Initial Company Information:
            {state["initial_documents"]}
            
            Ensure questions are clear, specific, and well-aligned with the company's context.
            """
            
            # Use LLM to generate sub-questions
            messages = ["system","Your task is to generate sub-questions based on the initial search results.",
                ("human",f"{prompt}")]

            sub_questions = await self.model.with_structured_output(TavilySearchInput).ainvoke(messages)
            
        except Exception as e:
            msg = f"An error occurred during sub-question generation: {str(e)}"
            return {"messages": [AIMessage(content=msg)], "sub_questions": None, "initial_documents": state['initial_documents']}
            
        
        return {"messages": [AIMessage(content=msg)], "sub_questions": sub_questions, "initial_documents": state['initial_documents']}
            
    async def run(self, state: ResearchState):
        result = await self.generate_sub_questions(state)
        return result