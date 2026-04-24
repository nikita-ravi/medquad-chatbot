import os
import xml.etree.ElementTree as ET
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

def load_medquad_data(base_dir: str):
    """
    Parses the MedQuAD XML files and extracts QA pairs into LlamaIndex Document objects.
    """
    documents = []
    
    # Iterate through all subdirectories
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.xml'):
                file_path = os.path.join(root, file)
                try:
                    tree = ET.parse(file_path)
                    root_elem = tree.getroot()
                    
                    # Get general document metadata
                    doc_id = root_elem.attrib.get('id', '')
                    source = root_elem.attrib.get('source', '')
                    url = root_elem.attrib.get('url', '')
                    focus = root_elem.findtext('Focus', default='')
                    
                    # Extract QA Pairs
                    qa_pairs = root_elem.find('QAPairs')
                    if qa_pairs is not None:
                        for qa_pair in qa_pairs.findall('QAPair'):
                            question_elem = qa_pair.find('Question')
                            answer_elem = qa_pair.find('Answer')
                            
                            if question_elem is not None and answer_elem is not None:
                                q_text = question_elem.text
                                a_text = answer_elem.text
                                q_type = question_elem.attrib.get('qtype', '')
                                
                                if not a_text or a_text.strip().lower() == "none" or len(a_text.strip()) < 5:
                                    continue
                                
                                # Combine Q and A for the document text
                                text = f"Question: {q_text}\nAnswer: {a_text}"
                                
                                # Create meta data for filtering and retrieval
                                metadata = {
                                    'doc_id': doc_id,
                                    'source': source,
                                    'url': url,
                                    'focus': focus,
                                    'question_type': q_type
                                }
                                
                                doc = Document(text=text, metadata=metadata)
                                documents.append(doc)
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")
                    
    print(f"Loaded {len(documents)} QA pairs from MedQuAD.")
    return documents

if __name__ == "__main__":
    db_path = "/Users/jagan/Downloads/application/rag_system/MedQuAD"
    docs = load_medquad_data(db_path)
    
    # Preview
    print("\nPreview of first document:")
    print("Metadata:", docs[0].metadata)
    print("Text snippet:", docs[0].text[:200], "...")
