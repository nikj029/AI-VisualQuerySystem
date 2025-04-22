import streamlit as st
import os
import google.generativeai as genai
from py2neo import Graph, Node, Relationship
import PIL.Image
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

os.environ["STREAMLI_SERVER_DEBUG"]="1"

# Set Google Vision API credentials
os.environ["Google_credential"] = "C:/Users/User/Desktop/"your-gen-lang-client.json""

# Configure Google Gemini API key
genai.configure(api_key="your_api_key")

# Configure the Gemini model
generation_config = {
    "temperature": 0.5,
    "top_p": 0.8,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

# Connect to Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "your_password"))

# Save uploaded files locally
def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary directory and return its path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name

# Upload image to Google Gemini
def upload_to_gemini(image_path):
    """Uploads the image file to Gemini and returns the file object."""
    file = genai.upload_file(image_path)
    return file

# Extract relationships using Google Gemini
def extract_relationships_from_gemini(prompt, *images):
    """Extract relationships from images using Google Gemini."""
    response = model.generate_content([
        prompt,
        *images,
    ])
    return response.text

# Store relationships in Neo4j
def store_relationships_in_neo4j(name, relationships):
    """Store extracted relationships in Neo4j with an associated name/ID."""
    for rel in relationships.split("\n"):
        try:
            source, target_data = rel.split("->")
            target, data = target_data.split(":")
            
            # Create nodes and relationships
            source_node = Node("Module", name=source.strip())
            target_node = Node("Module", name=target.strip())
            relation = Relationship(source_node, "SENDS", target_node, data=data.strip())
            
            # Attach a unique name/ID to the relationship context
            source_node["context_name"] = name
            target_node["context_name"] = name
            
            graph.merge(source_node, "Module", "name")
            graph.merge(target_node, "Module", "name")
            graph.merge(relation)
        except ValueError:
            continue

# Retrieve relevant context using TF-IDF
def retrieve_relevant_context(user_query, name, top_k=3):
    """Retrieve the most relevant relationships for the query using TF-IDF."""
    query = f"""
    MATCH (source:Module)-[rel:SENDS]->(target:Module)
    WHERE source.context_name = '{name}' OR target.context_name = '{name}'
    RETURN source.name AS source, target.name AS target, rel.data AS data
    """
    results = graph.run(query).data()

    # Combine the relationships into a list of context strings
    context_strings = [
        f"{r['source']} -> {r['target']}: {r['data']}" for r in results
    ]

    if not context_strings:
        return ""

    # Compute relevance scores using TF-IDF
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(context_strings + [user_query])
    query_vector = vectors[-1]  # Last vector is the user query
    scores = np.dot(vectors[:-1], query_vector.T).toarray().ravel()

    # Get top-k most relevant context
    top_indices = np.argsort(scores)[::-1][:top_k]
    relevant_context = "\n".join([context_strings[i] for i in top_indices])

    return relevant_context

# Query Gemini with augmented context
def query_gemini_with_context(user_query, context):
    """Generate a response from Google Gemini using augmented context."""
    prompt = f"""
    You are an AI trained to answer questions about relationships in diagrams. Use the following context to answer the user's query.

    Context:
    {context}

    User Query:
    {user_query}
    """
    response = model.generate_content([prompt])
    return response.text

# Streamlit app
st.title("AI-Powered Diagram QA System with RAG")
st.sidebar.title("Upload and Query System")
st.sidebar.write("Upload architecture or data flow diagrams and assign a name or ID for history-based querying.")

uploaded_files = st.sidebar.file_uploader(
    "Upload Flow Diagrams", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    st.write("### Uploaded Diagrams")
    local_image_paths = []  # Store local paths for uploaded files

    for uploaded_file in uploaded_files:
        # Save the file locally
        local_path = save_uploaded_file(uploaded_file)
        local_image_paths.append(local_path)

        # Display the uploaded image
        image = PIL.Image.open(local_path)
        st.image(image, caption=uploaded_file.name, use_column_width=True)

    # Prompt the user to assign a name or ID
    image_name = st.text_input("Assign a name or ID to this image set:", key="image_name")
    if image_name and st.button("Process Images"):
        try:
            # Upload images to Gemini
            gemini_files = [upload_to_gemini(path) for path in local_image_paths]
            
            # Extract relationships
            prompt = "Extract relationships from these diagrams in the format: 'Source -> Target: Data'"
            relationships = extract_relationships_from_gemini(prompt, *gemini_files)

            st.text("Extracted Relationships:\n" + relationships)

            # Store relationships in Neo4j
            store_relationships_in_neo4j(image_name, relationships)
            st.success(f"Relationships added to the knowledge graph with name/ID '{image_name}'.")
        except Exception as e:
            st.error(f"Error processing diagrams: {e}")

# Query Section
st.write("### Query the System")
query_name = st.text_input("Enter the name or ID of the image set for querying:", key="query_name")
user_query = st.text_area("Enter your query about the relationships:")

if query_name and user_query and st.button("Ask Query"):
    try:
        # Retrieve the most relevant context for the query
        context = retrieve_relevant_context(user_query, query_name, top_k=3)

        if context:
            # Query Gemini with relevant context
            response = query_gemini_with_context(user_query, context)
            st.write("### Gemini's Response:")
            st.write(response)
        else:
            st.warning(f"No relevant context found for the name/ID '{query_name}'.")
    except Exception as e:
        st.error(f"Error processing your query: {e}")
