from langchain_core.messages import AIMessage
from langchain_anthropic import ChatAnthropic

from ..classes import ResearchState,DocumentClusters



class ClusterNode:
    def __init__(self):
        self.model = ChatAnthropic(
            model="claude-haiku-4-5-20251001"
        )

    async def cluster(self, state: ResearchState):
        company = state['company']
        company_url = state['company_url']
        initial_docs = state['initial_documents']
        documents = state.get('documents', {})
   
        # Extract compnay domain from URL
        target_domain = company_url.split("//")[-1].split("/")[0]

        # Collect all retrieved documents without duplicates
        unique_urls = []
        seen_urls = set()
        for url, doc, in documents.items():
            if url not in seen_urls:
                unique_urls.append({'url': url, 'content': doc.get('content', '')})
                seen_urls.add(url)

        # Pass in the first 25 URLs
        urls = unique_urls[:25]

        # LLM prompt to categorize documents accurately
        prompt = f"""
            We conducted a search for a company called '{company}', but the results may include documents from other companies with similar names or domains.
            Your task is to accurately categorize these retrieved documents based on which specific company they pertain to, using the initial company information as "ground truth."

            ### Target Company Information
            - **Company Name**: '{company}'
            - **Primary Domain**: '{target_domain}'
            - **Initial Context (Ground Truth)**: Information below should act as a verification baseline. Use it to confirm that the document content aligns directly with {company}.
            - **{initial_docs}**

            ### Retrieved Documents for Clustering
            Below are the retrieved documents, including URLs and brief content snippets:
            {[{'url': doc['url'], 'snippet': doc['content']} for doc in urls]}

            ### Clustering Instructions
            - **Primary Domain Priority**: Documents with URLs containing '{target_domain}' should be prioritized for the main cluster for '{company}'.
            - **Include Relevant Third-Party Sources**: Documents from third-party domains (e.g., news sites, industry reports) should also be included in the '{company}' cluster if they provide specific information about '{company}', reference '{target_domain}', or closely match the initial company context.
            - **Separate Similar But Distinct Domains**: Documents from similar but distinct domains (e.g., '{target_domain.replace('.com', '.io')}') should be placed in separate clusters unless they explicitly reference the target domain and align with the company's context.
            - **Handle Ambiguities Separately**: Documents that lack clear alignment with '{company}' should be placed in an "Ambiguous" cluster for further review.

            ### Example Output Format
            {{
                "clusters": [
                    {{
                        "company_name": "Name of Company A",
                        "cluster": [
                            "http://example.com/doc1",
                            "http://example.com/doc2"
                        ]
                    }},
                    {{
                        "company_name": "Name of Company B",
                        "cluster": [
                            "http://example.com/doc3"
                        ]
                    }},
                    {{
                        "company_name": "Ambiguous",
                        "cluster": [
                            "http://example.com/doc4"
                        ]
                    }}
                ]
            }}

            ### Key Points
            - **Focus on Relevant Content**: Documents that contain relevant references to '{company}' (even from third-party domains) should be clustered with '{company}' if they align well with the initial information and context provided.
            - **Identify Ambiguities**: Any documents without clear relevance to '{company}' should be placed in the "Ambiguous" cluster for manual review.
        """

        # LLM call with structured output using DocumentClusters
        messages = ["system","Your job is to generate clusters for the company: '{company}'.\n",
                ("human",f"{prompt}")]
        
        msg = ""
        try:
            # Use the model's structured output with DocumentClusters format
            response = await self.model.with_structured_output(DocumentClusters).ainvoke(messages)
            clusters = response.clusters  # Access the structured clusters directly
      
        except Exception as e:
            msg = f"Error: {str(e)}\n"
            clusters = []


        # Summarize the results
        if not clusters:
            msg += "No valid clusters generated. Please check the document formats.\n"
        else:
            msg += "Clusters generated successfully:\n"
            urls = set()
            for  idx, cluster in enumerate(clusters, start=1):
                msg += f"   📂 Company {idx}: {cluster.company_name}\n"
                for url in cluster.cluster:
                    domain = url.split("://")[-1].split("/")[0]
                    if domain not in urls:
                        urls.add(domain)
                        msg += f"       📄 {domain}\n"
        
        return {"messages": [AIMessage(content=msg)], "document_clusters": clusters}
    
    # Define the function to choose the correct cluster as a conditional edge
    async def choose_cluster(self, state: ResearchState):
        company_url = state['company_url']
        clusters = state['document_clusters']

        # Attempt to automatically choose the correct cluster
        for index,cluster in enumerate(clusters):
            # Check if any URL in the cluster starts with the company URL
            if any(url.startswith(company_url) for url in cluster.cluster):
                # state['chosen_cluster'] = index
                msg = f"Automatically selected cluster for '{company_url}' as {cluster.company_name}."
                return {"messages": [AIMessage(content=msg)], "chosen_cluster": index}

        # If no automatic match, indicate that manual selection is needed
        msg = "No automatic cluster match found. Please select the correct cluster manually."
        return {"messages": [AIMessage(content=msg)], "document_clusters": clusters, "chosen_cluster": None}

    async def run(self, state: ResearchState, websocket):
        if websocket:
            await websocket.send_text("🔄 Beginning clustering process...")

        cluster_result = await self.cluster(state)
        state['document_clusters'] = cluster_result['document_clusters'] 
        choose_cluster_result = await self.choose_cluster(state)
        result = {'chosen_cluster': choose_cluster_result['chosen_cluster']}
        result.update(cluster_result)
        return result