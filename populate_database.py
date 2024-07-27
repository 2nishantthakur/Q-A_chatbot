'''
latest version(25/08/24)
user sessions working(not interfaring with different users)
headings are working fine now
'''
import os
import sys
import io
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import fitz

CHROMA_PATH = "chroma"
DATA_PATH = "PDFs"

def main(reset=False, db=None, username = None):
    print("Populate_database_UI_main function called")
    print(reset)
    if reset:
        print("✨ Clearing Database")
        clear_database()
    else:return

    # Ensure Chroma directory exists
    if not os.path.exists(CHROMA_PATH+"/"+username):
        os.makedirs(CHROMA_PATH+"/"+username)
    # settings = Settings(tenant="default_tenant", database="default_database")
    # db = Chroma(persist_directory=CHROMA_PATH+"_"+username, embedding_function=get_embedding_function(), settings = settings)
    db = Chroma(persist_directory=CHROMA_PATH+"_"+username, embedding_function=get_embedding_function())
    db.persist()
    # Create (or update) the data store.
    documents = load_documents(DATA_PATH+"/"+username)
    documents+=extract_text_from_images_in_pdf(DATA_PATH+"/"+username)
    chunks = split_documents(documents)
    add_to_chroma(chunks, db)

def load_documents(DATA_PATH):
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# def extract_text_from_images(DATA_PATH):
#     image_documents = []
#     for pdf_file in os.listdir(DATA_PATH):
#         if pdf_file.endswith(".pdf"):
#             pdf_path = os.path.join(DATA_PATH, pdf_file)
#             images = convert_from_path(pdf_path)
#             for i, image in enumerate(images):
#                 text = pytesseract.image_to_string(image)

                
#                 if text.strip():  # Only add if there's some text extracted
#                     print("\n\n\n\n\n"+text+"\n\n\n\n\n")
#                     metadata = {"source": pdf_file, "page": i+1}
#                     image_documents.append(Document(page_content=text, metadata=metadata))
#                     print(image_documents)
#     return image_documents

def extract_text_from_images_in_pdf(pdf_path):
    # Open the PDF document
    image_documents = []
    for doc in os.listdir(pdf_path):
        pdf_document = fitz.open(pdf_path+"/"+doc)

        # Loop through each page
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            image_list = page.get_images(full=True)

            # Loop through each image in the page
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]

                # Load the image using PIL
                image = Image.open(io.BytesIO(image_bytes))

                # Use Tesseract to extract text from the image
                text = pytesseract.image_to_string(image).strip()
                if text.strip():  # Only add if there's some text extracted
                    # print("\n\n\n\n\n"+text+"\n\n\n\n\n")
                    metadata = {"source": os.path.basename(pdf_path), "page": page_number}
                    image_documents.append(Document(page_content=text.strip(), metadata=metadata))
                    # print(image_documents)

    return image_documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document], db):
    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")


    #comment these two lines if you want the system to retain previous uploaded PDFs data.
    for doc_id in existing_ids:   
        db.delete(doc_id)

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]

        total_chunks = len(new_chunks)
        for i, chunk in enumerate(new_chunks, start=1):
            db.add_documents([chunk], ids=[chunk.metadata["id"]])
            progress_percentage = (i / total_chunks) * 100
            print(f"Progress: {progress_percentage:.2f}% ({i}/{total_chunks} chunks added)", end='\r')

        sys.stdout.flush()
        db.persist()
        print("✅ All new documents have been added and persisted")
    else:
        print("✅ No new documents to add")

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

if __name__ == "__main__":
    main()
