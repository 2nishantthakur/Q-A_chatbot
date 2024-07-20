import os
import sys
import io
import shutil
import logging
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import fitz

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the tesseract_cmd path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

CHROMA_PATH = "chroma"
DATA_PATH = "PDFs"

def main(reset=False, db=None):
    logger.info("Populate_database_UI_main function called")
    if reset:
        logger.info("✨ Clearing Database")
        clear_database()

    # Ensure Chroma directory exists
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
        client_settings=Settings()
    )

    # Create (or update) the data store.
    documents = load_documents(DATA_PATH)
    documents += extract_text_from_images_in_pdf(DATA_PATH)
    chunks = split_documents(documents)
    add_to_chroma(chunks, db)

def load_documents(data_path):
    document_loader = PyPDFDirectoryLoader(data_path)
    return document_loader.load()

def extract_text_from_images_in_pdf(pdf_path):
    image_documents = []
    for doc in os.listdir(pdf_path):
        pdf_document = fitz.open(os.path.join(pdf_path, doc))

        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]

                image = Image.open(io.BytesIO(image_bytes))
                text = pytesseract.image_to_string(image).strip()

                if text:
                    metadata = {"source": doc, "page": page_number + 1}
                    image_documents.append(Document(page_content=text, metadata=metadata))
                    logger.info(f"Extracted text from {doc}, page {page_number + 1}")

    return image_documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks, db):
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    logger.info(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    if new_chunks:
        logger.info(f"👉 Adding new documents: {len(new_chunks)}")

        total_chunks = len(new_chunks)
        for i, chunk in enumerate(new_chunks, start=1):
            db.add_documents([chunk], ids=[chunk.metadata["id"]])
            progress_percentage = (i / total_chunks) * 100
            logger.info(f"Progress: {progress_percentage:.2f}% ({i}/{total_chunks} chunks added)")

        db.persist()
        logger.info("✅ All new documents have been added and persisted")
    else:
        logger.info("✅ No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        logger.info("Database cleared")

if __name__ == "__main__":
    main()
