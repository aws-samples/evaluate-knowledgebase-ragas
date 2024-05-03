import argparse
import boto3
from datasets import Dataset
from langchain.chains import RetrievalQA
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
import logging
import pandas
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    faithfulness,
    context_recall,
    answer_relevancy,
)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
parser = argparse.ArgumentParser(description="Help", 
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--region", help="aws region", required=True)
parser.add_argument("--model-id", help="Id of the model to call, e.g., anthropic.claude-instant-v1", required=True)
parser.add_argument("--embedding-model-id", help="Id of the embedding model to call, e.g., amazon.titan-embed-text-v1", required=True)
parser.add_argument("--knowledgebase-id", help="Id of the Knowledge Base", required=True)
parser.add_argument("--evalresult-path", help="Path to the evaluation result file", required=True)
parser.add_argument("--testplan-path", help="Path to the testplan file", required=True)
args = parser.parse_args()

REGION = args.region
BEDROCK_ENDPOINT_URL = f"https://bedrock-runtime.{REGION}.amazonaws.com"
BEDROCK_AGENT_ENDPOINT_URL = f"https://bedrock-agent-runtime.{REGION}.amazonaws.com"

BEDROCK_EMBEDDING_MODEL_ID = args.embedding_model_id
BEDROCK_KNOWLEDGEBASE_ID = args.knowledgebase_id
BEDROCK_MODEL_ID = args.model_id
EVALRESULT_PATH = args.evalresult_path
TESTPLAN_PATH = args.testplan_path

# initialize bedrock runtime client
bedrock_client = boto3.client("bedrock-runtime", REGION)
model_kwargs = {"temperature": 0.4}
bedrock_model = BedrockChat(
    endpoint_url=BEDROCK_ENDPOINT_URL,
    region_name=REGION,
    model_id=BEDROCK_MODEL_ID,
    model_kwargs=model_kwargs,
)

# initialize bedrock embedding
bedrock_embeddings = BedrockEmbeddings(    
    region_name=REGION, model_id=BEDROCK_EMBEDDING_MODEL_ID
)

# initialize bedrock agent runtime client
bedrock_agent_client = boto3.client("bedrock-agent-runtime", REGION)

# initialize knowledgebase retriever
numberOfResults = 4
retriever = AmazonKnowledgeBasesRetriever(
    endpoint_url=BEDROCK_AGENT_ENDPOINT_URL,
    knowledge_base_id=BEDROCK_KNOWLEDGEBASE_ID,
    retrieval_config={
        "vectorSearchConfiguration": {"numberOfResults": numberOfResults}
    },
    client=bedrock_agent_client,
)

qa_chain = RetrievalQA.from_chain_type(
    llm=bedrock_model, retriever=retriever, return_source_documents=True
)

def generate_answer_context_with_knowledgebase(df):
    logger.info(f"Generating answer and context for question {df['question']}")
    question = df["question"]
    response = qa_chain({"query": question})
    source_documents = response["source_documents"]
    contexts_list_raw = list(map(lambda x: x.page_content, source_documents))
    df["contexts"] = str(contexts_list_raw)
    df["answer"] = response["result"]
    return df


# Read test plan and drop rows with empty columns
csv_test_df = pandas.read_csv(TESTPLAN_PATH)
filtered_csv_test_df = csv_test_df.dropna()

# Generate answers and context with knowledgebase based on test plan questions
logger.info(f"Starting generation of answers and context based on test plan")
final_csv_test_df = filtered_csv_test_df.apply(
    lambda x: generate_answer_context_with_knowledgebase(x), axis=1
)
final_csv_test_df["contexts"] = final_csv_test_df["contexts"].apply(
    lambda x: eval(x) if isinstance(x, str) else []
)
logger.info(f"Generation completed")
# Evaluate KnowledgeBase on a defined list of metrics
logger.info(f"Starting evaluation:")
metrics = [
    faithfulness,
    context_recall,
    context_precision,
    answer_relevancy,
]

csv_dataset = Dataset.from_pandas(final_csv_test_df)
result = evaluate(
    csv_dataset,
    metrics=metrics,
    llm=bedrock_model,
    embeddings=bedrock_embeddings,
)
logger.info(f"Evaluation completed: {result}")
df = result.to_pandas()
filtered_df = df.dropna()
logger.info(f"Detailed results: {filtered_df}")

# Save evaluation results in CSV file
df.to_csv(EVALRESULT_PATH)
logger.info(f"Evaluation results saved at: {EVALRESULT_PATH}")
