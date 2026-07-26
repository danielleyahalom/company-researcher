from datetime import datetime
from langchain_core.messages import AIMessage
from langchain_anthropic import ChatAnthropic
from ..classes import ResearchState



class GenerateNode:
    def __init__(self):
        self.model = ChatAnthropic(
            model="claude-sonnet-5",
            max_tokens=8192
        )
    def extract_text(self, content):
        # content can be a plain string or a list of content blocks (e.g. thinking + text)
        if isinstance(content, str):
            return content
        return "".join(
            block.get("text", "") for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )

    def extract_markdown_content(self, content):
    # Strip out extra preamble or conversational text, retaining only Markdown.
        start_index_hash = content.find("#")
        start_index_bold = content.find("**")
        
        if start_index_hash != -1 and (start_index_bold == -1 or start_index_hash < start_index_bold):
            # '#' found and it comes before '**' (or '**' not found)
            return content[start_index_hash:].strip()
        elif start_index_bold != -1:
            # '**' found
            return content[start_index_bold:].strip()
        else:
            # Neither '#' nor '**' found, return the whole content stripped
            return content.strip()

    async def generate_report(self, state: ResearchState):
        report_title = f"Weekly Report on {state['company']}"
        report_date = datetime.now().strftime('%B %d, %Y')

        prompt = f"""
        You are an expert researcher tasked with writing a fact-based report on recent developments for the company **{state['company']}**. Write the report in Markdown format. The title and date are added separately, so do not include a title, heading, or company name line of any kind before the first section — your response must begin directly with the "### Executive Summary" section heading. Each section must be written in well-structured paragraphs, not lists or bullet points.

        Be concise and to the point. Keep each section to 2-3 tight paragraphs at most, prioritizing the most important, information-dense facts over exhaustive coverage. Avoid filler, repetition, and generic statements that don't convey specific information about the company.

        Ensure the report includes:
        - **Inline citations** as Markdown hyperlinks directly in the main sections (e.g., Company X is an innovative leader in AI ([LinkedIn](https://linkedin.com))).
        - A **Citations Section** at the end that lists all URLs used.

        ### Report Structure:
        Use a `###` Markdown heading for each of the following six section titles.

        1. ### Executive Summary
            - High-level overview of the company, its services, location, employee count, and achievements.
            - Make sure to include the general information necessary to understand the company well including any notable achievements.

        2. ### Leadership and Vision
            - Details on the CEO and key team members, their experience, and alignment with company goals.
            - Any personnel changes and their strategic impact.

        3. ### Product and Service Overview
            - Summary of current products/services, features, updates, and market fit.
            - Include details from the company's website, tools, or new integrations.

        4. ### Financial Performance
            - For public companies: key metrics (e.g., revenue, market cap).
            - For startups: funding rounds, investors, and milestones.

        5. ### Recent Developments
            - New product enhancements, partnerships, competitive moves, or market entries.

        6. ### Citations
            - List each unique domain referenced in the report exactly once — if multiple articles from the same domain were used, list that domain only once, not once per article.
            - Format each entry as a Markdown hyperlink whose link text is just the domain name (e.g., [tavily.com](https://www.tavily.com/)), not the article title.

        ### Documents to Base the Report On:
        {state['documents']}
        """

        messages = [("system", "Your task is to generate a Markdown report."), ("human", prompt)]

        try:
            # Invoke the model
            response = await self.model.ainvoke(messages)

            # Extract the Markdown content
            markdown_content = self.extract_markdown_content(self.extract_text(response.content))

            # Add the title and date to the response
            full_report = f"# {report_title}\n\n*{report_date}*\n\n{markdown_content}"
            return {"messages": [AIMessage(content=f"Report generated successfully!\n{full_report}")], "report": full_report}
        except Exception as e:
            error_message = f"Error generating report: {str(e)}"
            return {
                "messages": [AIMessage(content=error_message)],
                "report": f"# Error Generating Report\n\n*{report_date}*\n\n{error_message}"
            }


    async def run(self, state: ResearchState, websocket):
        if websocket:
            await websocket.send_text("⌛️ Generating report...")
        result = await self.generate_report(state)
        return result