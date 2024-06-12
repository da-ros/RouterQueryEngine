# Agentic RAG with Router Query Engine

This project demonstrates how to build an agentic Retrieval-Augmented Generation (RAG) system using LlamaIndex and a router query engine. The system is designed to summarize and retrieve specific context from a given document, such as the MetaGPT paper.

## Setup

To get started with this project, follow the setup instructions below.

### Prerequisites

- Python 3.12+
- `wget` (for downloading the sample document)
- `pip` (Python package installer)

### Installation

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/da-ros/RouterQueryEngine.git
    cd RouterQueryEngine
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have your OpenAI API key available. You can store it in an environment variable or modify the helper function to return your API key directly.

## Load Data

Download the MetaGPT paper using the following command:

```bash
wget "https://openreview.net/pdf?id=VtmBAGCN7o" -O metagpt.pdf
```

## Usage

1. Load the documents:

    ```python
    from llama_index.core import SimpleDirectoryReader

    # Load documents
    documents = SimpleDirectoryReader(input_files=["metagpt.pdf"]).load_data()
    ```

2. Define the LLM and Embedding model:

    ```python
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    Settings.llm = OpenAI(model="gpt-3.5-turbo")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    ```

3. Create Summary Index and Vector Index:

    ```python
    from llama_index.core import SummaryIndex, VectorStoreIndex

    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes)
    ```

4. Define Query Engines and Set Metadata:

    ```python
    from llama_index.core.tools import QueryEngineTool

    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    vector_query_engine = vector_index.as_query_engine()

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description="Useful for summarization questions related to MetaGPT",
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description="Useful for retrieving specific context from the MetaGPT paper.",
    )
    ```

5. Define the Router Query Engine:

    ```python
    from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
    from llama_index.core.selectors import LLMSingleSelector

    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[summary_tool, vector_tool],
        verbose=True
    )
    ```

6. Query the Router Query Engine:

    ```python
    response = query_engine.query("What is the summary of the document?")
    print(str(response))

    response = query_engine.query("How do agents share information with other agents?")
    print(str(response))
    ```

## Putting Everything Together

For a complete example, see the provided script `L1_Router_Engine.py`. This script integrates all the steps above and demonstrates how to query the router query engine with specific questions about the MetaGPT paper.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for providing the language and embedding models.
- The authors of the MetaGPT paper for their valuable research.

---

For any questions or issues, please open an issue on the repository or contact the project maintainer.
